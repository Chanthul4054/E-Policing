from functools import wraps
from flask import abort
from flask_login import current_user, login_required

def admin_required(f):
    @wraps(f)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            abort(401)

        if current_user.role != "admin":
            abort(403)  
        return f(*args, **kwargs)
    return wrapper
