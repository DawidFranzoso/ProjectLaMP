from datetime import datetime

from DataDownloader import DataDownloader
from eval.evaluation import LaMPEvaluation
import json

DataDownloader.maybe_download_all()

golds_zip = ...
temp_dir = ...
pred_zip = ...
output_file = f"out{datetime.now().replace(microsecond=0)}"

evaluator = LaMPEvaluation(all_golds_zip_file_addr=golds_zip, extract_addr=temp_dir)
results = evaluator.evaluate_all(pred_zip)
with open(output_file, "w") as file:
    json.dump(results, file)