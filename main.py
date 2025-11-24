from datetime import datetime

from DataDownloader import DataDownloader
from eval.evaluation import LaMPEvaluation
import json

DataDownloader.maybe_download_all(tasks=[7], force=True)

x, y = [
    DataDownloader.DataEntry(
        task_number=7,
        split_name="validation",
        is_time_based=False,
        is_output=is_output,
    ).load_json() for is_output in (False, True)
]

print("X:")
print(str(x)[-7000:])
print("Y:")
print(str(y)[-7000:])

"""

output_file = f"out{datetime.now().replace(microsecond=0)}"



evaluator = LaMPEvaluation(single_gold_json_file_addr=golds_zip)

for task in range(1, 1 + 7):
    json_filepath = DataDownloader.DataEntry()

    results = evaluator.evaluate_task(predicts_json_addr=..., "")
    with open(output_file, "w") as file:
        json.dump(results, file)

"""