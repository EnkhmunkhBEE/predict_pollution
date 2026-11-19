
import json

from predictor import predict_by_date


result = predict_by_date(
    "2025-01-01",
    "2027-06-01"
)

print(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=4
    )
)
