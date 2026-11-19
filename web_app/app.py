
from flask import Flask, render_template, request, jsonify
from predictor import predict_by_date

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")





@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        start_date = data.get("start_date")
        end_date = data.get("end_date")

        result = predict_by_date(
            start_date,
            end_date
        )

        return jsonify(result)

    except ValueError as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
