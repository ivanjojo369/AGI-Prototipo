# task_manager.py
class TaskManager:
    def __init__(self):
        self._log = []

    def submit(self, task: str, **kwargs):
        entry = {"task": task, "kwargs": kwargs}
        self._log.append(entry)
        return {"status": "ok", "queued": entry}

    def history(self):
        return list(self._log)
