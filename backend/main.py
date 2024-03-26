# main.py
from flask import Flask, jsonify, request
import json
import datetime
import pymongo

class RecommendationModel:
    def __init__(self, user_data: dict, product_data: dict):
        self.user_data = user_data
        self.product_data = product_data

    def predict(self, user_id):
        user_preferences = self.user_data[user_id]
        user_sum_ratings = sum(user_preferences.values())
        user_count = len(user_preferences)

        product_scores = []
        for product, product_name in self.product_data.items():
            if product not in user_preferences.keys():
                average_rating = sum(user_preferences.values()) / user_count
                product_scores.append((product, average_rating))

        product_scores = sorted(product_scores, key=lambda x: x[1], reverse=True)
        return [product_name for product, _ in product_scores]

app = Flask(__name__)

# Connect to the database
app.config['MONGO_URI'] = 'mongodb+srv://<username>:<password>@cluster0.scqmoj3.mongodb.net/<database_name>'
mongo = pymongo.MongoClient(app.config['MONGO_URI'])

# Load example data from JSON
with open("example_data.json") as example_data_file:
    example_data = json.load(example_data_file)

# Create an instance of the RecommendationModel class
user_data = example_data["user_data"]
product_data = example_data["product_data"]

recommendation_model = RecommendationModel(user_data, product_data)

with app.app_context():

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
                'timestamp': datetime.datetime.now()
            }
        )

        # Get the recommended products
        recommended_products = recommendation_model.predict(user_id)

        return jsonify({'recommendations': recommended_products}), 200

    if __name__ == '__main__':
        app.run(debug=True)