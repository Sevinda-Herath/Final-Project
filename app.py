from flask import Flask, request, jsonify
from lstm_model import train_and_predict

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Missing ticker'}), 400

    try:
        predictions = train_and_predict(ticker)
        return jsonify({
            'ticker': ticker,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
