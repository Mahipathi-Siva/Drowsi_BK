from flask import Blueprint, request, jsonify
from database import users_collection
from routes.auth_routes import token_required

user = Blueprint("user", __name__)

@user.route("/profile", methods=["POST"])
@token_required
def profile(current_user):
    return jsonify({
        "user": {
            "name": current_user["name"],
            "email": current_user["email"],
            "phone_number": current_user["phone_number"],
            "license_number": current_user["license_number"],
            "vehicle_number": current_user["vehicle_number"],
            "Days":current_user["day"],
            "count":current_user["count"]
        }
    }), 200


@user.route("/edit_profile", methods=["PUT"])
@token_required
def edit_profile(current_user):
    try:
        data = request.json
        update_fields = {key: value for key, value in data.items() 
                         if key in ["name", "phone_number", "license_number", "vehicle_number"]}
        
        if not update_fields:
            return jsonify({"message": "No valid fields to update"}), 400
        
        # Use users_collection as a variable, not a function
        users_collection.update_one({"email": current_user["email"]}, {"$set": update_fields})
        
        # Return the updated profile
        updated_user = users_collection.find_one({"email": current_user["email"]})
        return jsonify({
            "message": "Profile updated successfully",
            "user": {
                "name": updated_user["name"],
                "email": updated_user["email"],
                "phone_number": updated_user["phone_number"],
                "license_number": updated_user["license_number"],
                "vehicle_number": updated_user["vehicle_number"]
            }
        }), 200
    except Exception as e:
        return jsonify({"message": f"Error updating profile: {str(e)}"}), 500