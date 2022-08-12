from flask import Flask, render_template,request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def home():
    if request.method=="POST":
        inp=request.form.get("inp")
        sidm= SentimentIntensityAnalyzer()
        score = sidm.polarity_scores(inp)
        if score["neg"]!=0:
            return render_template('index.html',message='Negative')
    return render_template('index.html',message='')
if __name__=="__main__":
    app.run(debug=True)