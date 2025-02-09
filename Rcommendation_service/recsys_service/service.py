from recommender import Recommender
from codecs import open
import time
from flask import Flask, request, jsonify
app = Flask(__name__)

print ("Preparing recommender")
start_time = time.time()
recommender_instance = Recommender()
print( "Recommender is ready")
print (time.time() - start_time, "seconds")

@app.route("/recommender", methods=["POST", "GET"])
def recommender():
    if request.method == "POST":
        data = request.json

        if data['model'] == 'most_popular_model': 
            recommendations = recommender_instance.get_most_popular_books_recommendations(data['user_id'])
        else: 
            recommendations = recommender_instance.get_hybrid_recommendations(data['user_id'], data['usr_ind'])

        response = {f'result':recommendations}

    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5509, debug=False)
