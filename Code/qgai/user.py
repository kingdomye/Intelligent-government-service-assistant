"""User session and asynchronous task primitives."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from . import classify, datamining, face, inquiry
from .lang import translate

__all__ = [
    "User",
    "TableFiller",
    "AsyncModel",
    "AsyncPredictLooper",
    "AsyncTrainLooper",
    "LoopBed",
]

DEFAULT_INFO_PATH = Path(__file__).with_name("user_info.json")
with DEFAULT_INFO_PATH.open(encoding="utf-8") as default_info_file:
    _default_info = json.load(default_info_file)

DEFAULT_NECESSARY_INFO: dict[str, Any] = _default_info["necessary"]
DEFAULT_ADDITIONAL_INFO: dict[str, Any] = _default_info["addition"]
DEFAULT_MAIN_INFO = {**DEFAULT_NECESSARY_INFO, **DEFAULT_ADDITIONAL_INFO}


class TableFiller:
    """Track and fill a sequence of government-service forms."""

    def __init__(self, tables: list[dict[str, Any]]):
        if not isinstance(tables, list):
            raise TypeError("tables must be a list")
        self._tables = tables
        self._pointer = 0

    def __setitem__(self, key: str, value: Any) -> None:
        self.table[key] = value

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._tables[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._tables)

    @property
    def table(self) -> dict[str, Any]:
        if self.is_finish:
            raise IndexError("all tables have been completed")
        return self._tables[self._pointer]

    @table.setter
    def table(self, table: dict[str, Any]) -> None:
        if self.is_finish:
            raise IndexError("all tables have been completed")
        self._tables[self._pointer] = table

    @property
    def tables(self) -> list[dict[str, Any]]:
        return self._tables

    @property
    def pointer(self) -> int:
        return self._pointer

    @property
    def is_finish(self) -> bool:
        return self._pointer >= len(self._tables)

    @property
    def bussiness_type(self) -> dict[str, int]:
        """Compatibility alias; prefer :mod:`qgai.classify`."""
        return classify.type_dic

    def reset_pointer(self) -> None:
        self._pointer = 0

    def next_table(self) -> dict[str, Any]:
        table = self.table
        self._pointer += 1
        return table


AsyncTask = Callable[..., Coroutine[Any, Any, Any]]


class AsyncModel:
    """Schedule one coroutine at a time on a background event loop."""

    processing = -2147483648

    def __init__(
        self,
        task: AsyncTask,
        loop: asyncio.AbstractEventLoop,
        wait_time: float = 5,
    ):
        self._future: Future[Any] | None = None
        self._task = task
        self._loop = loop
        self._history: list[Any] = []
        self._lock = threading.Lock()
        self.wait_time = wait_time

    def __call__(self, *args: Any) -> Any:
        return self.activate(*args)

    def __getitem__(self, index: int) -> Any | None:
        try:
            return self._history[index]
        except IndexError:
            return None

    def activate(self, *args: Any) -> Any:
        with self._lock:
            if self._future is None:
                if not self._loop.is_running():
                    raise RuntimeError("background event loop is not running")
                self._future = asyncio.run_coroutine_threadsafe(
                    self._task(*args),
                    self._loop,
                )
            if not self._future.done():
                return self.processing
            completed = self._future
            self._future = None

        value = completed.result()
        self._history.append(value)
        return value

    async def async_activate(self, *args: Any) -> Any:
        value = await self._task(*args)
        self._history.append(value)
        return value

    @property
    def done(self) -> bool:
        return self._future is None or self._future.done()

    def cancel(self) -> bool:
        return self._future.cancel() if self._future is not None else False


class LoopBed:
    """Own an asyncio event loop running on a daemon thread."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._loop.is_running()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def create_new_task(self, async_func: AsyncTask) -> AsyncModel:
        return AsyncModel(async_func, self._loop)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def looping_on_new_thread(self) -> "LoopBed":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._thread = threading.Thread(
            target=self._run,
            name="qgai-event-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("background event loop failed to start")
        return self

    def stop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    @staticmethod
    def nonblock(wait_time: float = 5):
        def decorate(func: AsyncTask) -> AsyncModel:
            return AsyncModel(func, asyncio.get_event_loop(), wait_time=wait_time)

        return decorate


class AsyncLooper:
    def __init__(self, func: AsyncTask):
        self._bed = LoopBed()
        self._async_models: dict[str, AsyncModel] = {}
        self._task = func

    def __getitem__(self, identifier: str) -> AsyncModel:
        return self._async_models[identifier]

    def __contains__(self, identifier: str) -> bool:
        return identifier in self._async_models

    def __delitem__(self, identifier: str) -> None:
        del self._async_models[identifier]

    def append(self, identifier: str) -> None:
        self._async_models[identifier] = self._bed.create_new_task(self._task)

    def run_loop(self) -> None:
        self._bed.looping_on_new_thread()

    def run_loop_on_new_thread(self) -> None:
        self.run_loop()


class AsyncPredictLooper(AsyncLooper):
    def __init__(self):
        super().__init__(self.async_predict_face)

    @staticmethod
    async def async_predict_face(imgs_bin: list[bytes]) -> str | None:
        return face.cv2_predict(imgs_bin)


class AsyncTrainLooper(AsyncLooper):
    def __init__(self):
        super().__init__(self.async_train_face)

    @staticmethod
    async def async_train_face(user_id: str, imgs_bin: list[bytes]) -> bool:
        return face.cv2_train(imgs_bin, user_id)


class User:
    def __init__(self, info_dic: dict[str, Any], loop_bed: LoopBed):
        self._data = datamining.DataMiningAgent()
        self._main_info = {**DEFAULT_MAIN_INFO, **info_dic}
        self._once_info: dict[str, Any] = {}
        self._label: int | None = None
        self._flow: str | None = None
        self._tables_filler: TableFiller | None = None
        self._loop_bed = loop_bed

        self._get_answer_res = loop_bed.create_new_task(self._get_answer)
        self._inquire_res = loop_bed.create_new_task(self._inquire)
        self._classify_res = loop_bed.create_new_task(self._classify)
        self._get_flow_res = loop_bed.create_new_task(self._get_flow)
        self._train_face_res = loop_bed.create_new_task(self._train_face)
        self._predict_face_res = loop_bed.create_new_task(self._predict_face)
        self._init_res = loop_bed.create_new_task(self._initialize)

    def __setitem__(self, key: str, value: Any) -> None:
        target = self._main_info if key in DEFAULT_MAIN_INFO else self._once_info
        target[key] = value

    @property
    def info(self) -> dict[str, Any]:
        combined = {**self._once_info, **self._main_info}
        return {(translate(key) or key): value for key, value in combined.items()}

    @property
    def main_info(self) -> dict[str, Any]:
        return self._main_info

    @property
    def once_info(self) -> dict[str, Any]:
        return self._once_info

    @property
    def label(self) -> int:
        if self._label is not None:
            return self._label
        previous = self.classify[-1]
        if previous is None:
            raise RuntimeError("classify must complete before reading label")
        return previous

    @property
    def bus_type(self) -> str:
        labels_by_id = {value: key for key, value in classify.type_dic.items()}
        try:
            return labels_by_id[self.label]
        except KeyError as exc:
            raise RuntimeError(f"unsupported business label: {self.label}") from exc

    @property
    def flow(self) -> str:
        if self._flow is not None:
            return self._flow
        previous = self.get_flow[-1]
        if previous is None:
            raise RuntimeError("get_flow must complete before reading flow")
        return previous

    @property
    def tables(self) -> list[dict[str, Any]]:
        return self.tables_filler.tables

    @tables.setter
    def tables(self, value: list[dict[str, Any]]) -> None:
        self._tables_filler = TableFiller(value)

    @property
    def tables_filler(self) -> TableFiller:
        if self._tables_filler is None:
            raise RuntimeError("user has not been initialized")
        return self._tables_filler

    def fill_in_table(self) -> None:
        current_table = self.tables_filler.table
        info = self.info
        for key, value in current_table.items():
            if value in (None, "") and key in info:
                self.tables_filler[key] = info[key]

    def export_tables(self) -> list[dict[str, Any]]:
        return [
            {
                "header": {key: key for key in table},
                "row": list(table.values()),
            }
            for table in self.tables
        ]

    async def _inquire(self):
        if self.tables_filler.is_finish:
            return None
        question = inquiry.inquire(self.tables_filler.table)
        if question is None:
            return None
        return next(iter(question.items()))

    @property
    def inquire(self) -> AsyncModel:
        return self._inquire_res

    def inquire_func(self):
        return self._inquire_res.activate()

    async def _get_answer(self, text: str, key: str) -> str:
        return inquiry.get_answer(text, key)

    @property
    def get_answer(self) -> AsyncModel:
        return self._get_answer_res

    def get_answer_func(self, text: str, key: str):
        return self._get_answer_res.activate(text, key)

    async def _classify(self, requirement: str) -> int | None:
        self._label = classify.classify(requirement, classify.type_dic)
        return self._label

    @property
    def classify(self) -> AsyncModel:
        return self._classify_res

    def classify_func(self, requirement: str):
        return self._classify_res.activate(requirement)

    async def _get_flow(self) -> str:
        stream = await self._data.get_flow(self.label, self.info)
        if stream is None:
            raise RuntimeError("flow generator is unavailable")
        chunks = [chunk async for chunk in stream]
        self._flow = "".join(chunks)
        return self._flow

    @property
    def get_flow(self) -> AsyncModel:
        return self._get_flow_res

    def get_flow_func(self):
        return self._get_flow_res.activate()

    async def _train_face(self, user_id: str, imgs_bin: list[bytes]) -> bool:
        return face.cv2_train(imgs_bin, user_id)

    @property
    def train_face(self) -> AsyncModel:
        return self._train_face_res

    def enter_face_func(self, user_id: str, imgs_bin: list[bytes]):
        return self._train_face_res.activate(user_id, imgs_bin)

    async def _predict_face(self, imgs_bin: list[bytes]) -> str | None:
        return face.cv2_predict(imgs_bin)

    @property
    def predict_face(self) -> AsyncModel:
        return self._predict_face_res

    def predict_face_func(self, imgs_bin: list[bytes]):
        return self._predict_face_res.activate(imgs_bin)

    async def qna_hosting(self, hoster: AsyncGenerator):
        while not self.tables_filler.is_finish:
            question_data = await self._inquire()
            if question_data is None:
                self.tables_filler.next_table()
                continue
            key, question = question_data
            await hoster.asend(question)
            answer = await hoster.__anext__()
            value = await self._get_answer(answer, key)
            self.tables_filler[key] = value
            self[key] = value
        await hoster.asend("")
        return True

    async def train_hosting(
        self,
        hoster: AsyncGenerator,
        user_id: str,
        need_num: int = 50,
    ) -> bool:
        face_images = []
        async for image in hoster:
            feature = face.face_fetcher(image)
            if feature is not None:
                face_images.append(feature)
            if len(face_images) >= need_num:
                success = await self._train_face(user_id, face_images)
                await hoster.asend("success" if success else "failed")
                return success
            await hoster.asend("continue")
        return False

    async def _initialize(self, requirement: str) -> tuple[int, str]:
        label = await self._classify(requirement)
        if label is None:
            raise ValueError("无法识别对应的政务服务类型")
        tables = self._data.get_tables(label)
        if tables == ["-1"]:
            raise ValueError(f"找不到业务 {label} 对应的表格")
        self.tables = tables
        flow = await self._get_flow()
        return label, flow

    @property
    def init(self) -> AsyncModel:
        return self._init_res
