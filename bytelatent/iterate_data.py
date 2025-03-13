import json

import pyarrow
import typer
from rich.progress import track

from bytelatent.data.iterators.multiprocess_iterator import MultiprocessIteratorState
from bytelatent.logger import init_logger


def main(state_file: str):
    init_logger()
    pyarrow.set_io_thread_count(2)
    pyarrow.set_cpu_count(2)
    with open(state_file) as f:
        train_state = json.load(f)
        dl_state = MultiprocessIteratorState(**train_state["data_loader_state"])
        packing_iterator_state = dl_state.base_iterator_state
        print("building")
        packing_iterator = packing_iterator_state.build()
        print("iter")
        batch_iter = packing_iterator.create_iter()
        batch = None
        print("looping")
        i = 0
        for i in track(range(3_000)):
            batch = next(batch_iter)
            if i % 100 == 0:
                print(pyarrow.default_memory_pool())
        print(i)
        print(pyarrow.default_memory_pool())


if __name__ == "__main__":
    typer.run(main)
