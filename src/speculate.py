from threading import Thread, Lock
from re import match
from os import cpu_count
from copy import copy, deepcopy
from uuid import uuid4
from itertools import cycle
from collections import namedtuple
from functools import wraps

from joblib import hash as joblib_hash
from dask.distributed import Client


def seval(args, pipeline, env, n_jobs=-1, client=None):
    """
    Run a pipeline using speculative evaluation.

    """

    if client is None:
        client = Client()

    if n_jobs == -1:
        n_jobs = cpu_count()

    try:
        scheduler = Scheduler(n_jobs, client)
        scheduler.start()
        trunk = _eval_steps(pipeline, args, env, scheduler, n_jobs)
        while True:
            next(trunk)

    except StopEval as se:
        return se.result

    finally:
        scheduler.stop()


# Run steps in the pipeline out-of-order until an exception is raised.
def _eval_steps(pipeline, args, env, scheduler, n_jobs, step_number=0):
    results = args[:]
    pending_jobs = []
    for step in cycle(pipeline):
        for _ in _wait_for_jobs(pipeline,
                                results,
                                env,
                                pending_jobs,
                                scheduler,
                                n_jobs,
                                step,
                                step_number):
            yield

        if step.star_args and len(results) >= step.n_args:
            # Submit all results for evaluation at once.
            pending_jobs.append(scheduler.submit(step.concurrent,
                                                 step.action,
                                                 results))

            results = []

        while len(results) >= step.n_args:
            # Submit results for evaluation in chunks.
            pending_jobs.append(scheduler.submit(step.concurrent,
                                                 step.action,
                                                 results[:step.n_args]))

            results = results[step.n_args:]

        if len(results) > 0:
            raise TypeError('n_args mismatch for pipeline step {step.name}: expected {n_args} args but got {len(results)}')

        yield


def _parse_args(n_args):
    if isinstance(n_args, int):
        return n_args, False

    if isinstance(n_args, str):
        # Example of a matched string: "3+"
        match_object = match(r'(\d+)\+$', n_args)
        if match_object:
            return int(match_object.group(1)), True

    raise ValueError('{step.n_args} not a valid value for n_args')


def _wait_for_jobs(pipeline,
                   results,
                   env,
                   pending_jobs,
                   scheduler,
                   n_jobs,
                   step,
                   step_number):
    # results that have been evaluated by speculative evaluation.
    speculated_args = []
    speculative_branches = []
    # Loop until all pending jobs have completed.
    while True:
        new_jobs = []
        # Separate pending jobs from completed jobs.
        for job in pending_jobs:
            if job.done:
                results.append(job.result())

            else:
                new_jobs.append(job)

        pending_jobs = new_jobs
        if not pending_jobs:
            # All jobs have completed - forget speculation branches
            # and exit the loop.
            scheduler.forget_children()
            return

        if speculative_branches:
            # Hand evaluation over to the first branch in the queue
            # and then move it to the back of the queue.
            branch = speculative_branches.pop(0)
            next(branch)
            speculative_branches.append(branch)
            yield

        if len(results) - len(speculated_args) >= step.n_args and n_jobs > len(scheduler._jobs):
            branch, speculated_args = _speculative_evaluation(step.star_args,
                                                              pipeline,
                                                              results,
                                                              env,
                                                              scheduler,
                                                              n_jobs,
                                                              step_number,
                                                              speculated_args)
            speculative_branches.append(branch)

        yield


def _speculative_evaluation(star_args,
                            pipeline,
                            results,
                            env,
                            scheduler,
                            n_jobs,
                            step_number,
                            speculated_args):
    # Perform speculative evaluation on the available results.
    if star_args:
        # Evaluate all available results.
        branch = _eval_steps(pipeline,
                             results,
                             deepcopy(env),
                             scheduler.branch(),
                             n_jobs,
                             step_number+1)

        speculated_args = results

    else:
        # Only evaluate results that haven't
        # previously been evaluated.
        results_subset = results[len(speculated_args):]
        branch = _eval_steps(pipeline,
                             results_subset,
                             deepcopy(env),
                             scheduler.branch(),
                             step_number+1)

        speculated_args.extend(results_subset)

    next(branch)
    return branch, speculated_args


class Scheduler(Thread):
    def __init__(self, n_jobs, client):
        self._n_jobs = n_jobs
        self._client = client
        self._root_node = uuid4()
        self._job_graph = {self._root_node: []}
        self._queue = []
        self._jobs = []
        self._terminate = False
        self._lock = Lock()
        super().__init__()

    def _traverse_job_graph(self, node):
        nodes = [node]
        for child_node in self._job_graph[node][1:]:
            if child_node in self._job_graph:
                nodes.extend(self._traverse_job_graph(child_node))

        return nodes

    def _node_depth(self, node, _depth=0):
        parent = self._job_graph[node][0]
        if parent is None:
            return _depth

        return self._node_depth(parent, _depth+1)

    def branch(self):
        self._lock.acquire()
        while True:
            node = uuid4()
            if node not in self._job_graph:
                break

        self._job_graph[node] = [self._root_node]
        self._job_graph[self._root_node].append(node)
        new_branch = copy(self)
        new_branch._root_node = node
        self._lock.release()

        return new_branch

    def submit(self, concurrent, func, *args):
        job = Job(self._root_node, concurrent, func, args)
        self._lock.acquire()
        self._queue.append(job)
        self._lock.release()

        return job

    def forget_children(self):
        self._lock.acquire()
        children = set(self._traverse_job_graph(self._root_node)[1:])
        self._queue = [p for p in self._queue if p._node not in children]
        self._lock.release()

    def cancel_all(self):
        self._lock.acquire()
        for job in self._jobs:
            job._aborted = True
            job._future.cancel()

        self._jobs = []
        self._lock.release()

    def stop(self):
        self._terminate = True
        self.cancel_all()
        self.join()

    def run(self):
        while True:
            if self._terminate:
                return

            self._lock.acquire()
            new_jobs = [job for job in self._jobs if not job.done]
            self._jobs = new_jobs
            self._queue.sort(key=lambda p: self._node_depth(p._node))
            if self._queue and self._n_jobs > len(self._jobs):
                job = self._queue.pop()
                if job._concurrent:
                    job._future = self._client.submit(job._func, *job._args)
                    self._jobs.append(job)

                else:
                    job._result = job._func(*job._args)
                    self._done = True

            elif self._queue:
                queue_prio = self._node_depth(self._queue[0].node)
                self._jobs.sort(key=lambda p: self._node_depth(p._node))
                job_to_abort = None
                for job in self._jobs:
                    job_prio = self._node_depth(job.node)
                    if job_prio > queue_prio:
                        job_to_abort = job

                if job_to_abort:
                    self._jobs.remove(job_to_abort)
                    job_to_abort.abort()

            self._lock.release()


class Job:
    def __init__(self, node, concurrent, func, args):
        self._node = node
        self._concurrent = concurrent
        self._func = func
        self._args = args
        self._future = None
        self._result = None
        self._done = False
        self._aborted = False

    @property
    def aborted(self):
        return self._aborted

    @property
    def done(self):
        if self._done:
            return True

        if self._future and self._future.done():
            return True

        return False

    @property
    def result(self):
        if self._result:
            return self._result

        if self._future and self._future.done():
            self._result = self._future.result()
            return self._result

        raise NotDoneError

    def abort(self):
        if self._future:
            self._future.cancel()

        self._aborted = True


class Pipeline:
    def __init__(self, steps):
        self.steps = []
        for step in steps:
            n_args, star_args = _parse_args(step[1])
            if n_args > 0:
                func = memoize(step[3])

            else:
                func = step[3]

            self.steps.append(Step(str(step[0]),
                                   n_args,
                                   star_args,
                                   bool(step[2]),
                                   func))


Step = namedtuple('Step',
                  ('name', 'n_args', 'star_args', 'concurrent', 'action'))


def memoize(func):
    cache = dict()

    @wraps(func)
    def memoized_func(*args):
        args_hash = joblib_hash(args)
        if args_hash in cache:
            return cache[args_hash]

        result = func(*args)
        cache[args_hash] = result
        return result

    return memoized_func


class StopEval(Exception):
    def __init__(self, result):
        self.result = result


class NotDoneError(Exception):
    pass
