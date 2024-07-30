import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, session

# Existing imports and configurations
from models import get_user_by_id, update_user_profile

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = get_user_by_id(user_id)

    if request.method == 'POST':
        new_username = request.form['username']
        new_password = request.form['password']

        # Handle file upload
        profile_image = user[4]  # Assume profile image is stored in user[4] by default

        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                profile_image = filename

        update_user_profile(user_id, new_username, new_password, profile_image)
        flash('Profile updated!', 'success')
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)
