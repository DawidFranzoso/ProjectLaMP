import json
import os
from dataclasses import dataclass
from typing import Literal, List
import requests


class DataDownloader:
    @dataclass
    class DataEntry:
        task_number: int
        split_name: Literal["training", "validation", "test"]
        is_time_based: bool
        is_output: bool

        def get_url(self):
            maybe_time = "/time" if self.is_time_based else ""
            task_name = f"LaMP_{self.task_number}" if self.task_number != 2 else "LaMP_2/new"
            url_split_name = {
                "training": "train",
                "validation": "dev",
                "test": "test",
            }[self.split_name]

            return (
                f"https://ciir.cs.umass.edu/downloads/LaMP{maybe_time}/{task_name}/{url_split_name}"
                f"/{url_split_name}_{'outputs' if self.is_output else 'questions'}.json"
            )

        def get_filename(self):
            return (
                f"data/download/task{self.task_number}/{'time' if self.is_time_based else 'user'}_based"
                f"/{self.split_name}/{'y' if self.is_output else 'x'}.json"
            )

        def load_json(self) -> dict:
            with open(self.get_filename()) as f:
                return json.load(f)

    @staticmethod
    def maybe_download_entry(entry: DataEntry, force: bool = False):
        filepath = entry.get_filename()
        filename = filepath.split("/")[-1]
        dirpath = "/".join(filepath.split("/")[:-1])

        os.makedirs(dirpath, exist_ok=True)

        if filename in os.listdir(dirpath) and not force:
            return

        print(f"Downloading: {filepath}")

        data_stream = requests.get(entry.get_url(), stream=True)

        if data_stream.encoding is None:
            data_stream.encoding = 'utf-8'

        with open(filepath, "w") as f:
            delimiter = "id"
            for line_no, line in enumerate(data_stream.iter_lines(decode_unicode=True, delimiter=delimiter)):
                if line_no != 0:
                    line = delimiter + line

                if (line_no + 1) % 100 == 0:
                    print(f"{line_no + 1} lines complete")
                f.write(line)

        print(f"Downloaded: {filepath}")

    @staticmethod
    def maybe_download_all(tasks: List[int] = None, force: bool = False):
        if tasks is None:
            tasks = list(range(1, 1 + 7))

        # noinspection PyTypeChecker
        [
            DataDownloader.maybe_download_entry(
                DataDownloader.DataEntry(
                    task_number=task_number,
                    split_name=split_name,
                    is_time_based=is_time_based,
                    is_output=is_output,
                ),
                force=force,
            )
            for task_number in tasks
            for split_name in ["training", "validation", "test"]
            for is_time_based in [False, True]
            for is_output in [False, True]
        ]
