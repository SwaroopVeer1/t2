from flask import Flask, request, render_template
from handler import generate_image
import base64

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    image_data = None
    prompt = ""
    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        if prompt:
            result = generate_image(prompt)
            image_data = result.get("image_base64")
    return render_template("index.html", image_data=image_data, prompt=prompt)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
