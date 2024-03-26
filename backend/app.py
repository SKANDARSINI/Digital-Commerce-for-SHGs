# from flask import Flask, jsonify

# app = Flask(__name__)

# # Define a route for the root URL
# @app.route('/')

# # Define a route for recommendations
# @app.route('/recommendations/<user_id>')
# def get_recommendations(user_id):
#     # Call machine learning logic to generate recommendations based on user_id
#     recommendations = generate_recommendations(1)
#     return jsonify(recommendations)

# # Placeholder function for generating recommendations
# def generate_recommendations(user_id):
#     # Implement your machine learning code here
#     # This function should return recommended products based on the provided user ID
#     return {'user_id': 1, 'recommendations': ['product1', 'product2', 'product3']}

# if __name__ == '__main__':
#     # Run the Flask app
#     app.run(debug=True)

from flask import Flask, jsonify, request
from ml.recommendation import RecommendationModel
import pymongo
import time

app = Flask(__name__)

# Connect to the database
app.config['MONGO_URI'] = 'mongodb+srv://Skandarsini:darsh%402312@cluster0.scqmoj3.mongodb.net/thrivetogether'
mongo = pymongo.MongoClient(app.config['MONGO_URI'])

# Create the Flask app
recommendation_model = RecommendationModel()



# Define the endpoint for adding recommendations
@app.route('/recommendations', methods=['POST'])
def add_recommendations():
    user_id = request.json['user_id']
    product_id = request.json['product_id']
    rating = request.json['rating']

    # Add the recommendation to the database
    db = mongo.db
    recommendations = db.recommendations
    recommendations.insert_one(
        {
            'user_id': user_id,
            'product_id': product_id,
            'rating': rating,
            'timestamp': int(time.time())
        }
    )

    # Get the recommended products
    recommended_products = recommendation_model.predict(user_id)

    return jsonify({'recommendations': recommended_products}), 200

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)