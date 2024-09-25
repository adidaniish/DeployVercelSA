import googleapiclient.discovery
from transformers import pipeline
import plotly.express as px
from collections import Counter
import os
from flask import Flask, request, render_template

app = Flask(__name__)

def fetch_youtube_comments(video_id, api_key, max_comments=100):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    comments = []
    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    ).execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=max_comments,
                textFormat="plainText"
            ).execute()
        else:
            break

    return comments

# Initialize sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis')

def analyze_sentiment(comments):
    sentiment_results = []
    for comment in comments:
        result = sentiment_analyzer(comment)
        sentiment_results.append({
            'comment': comment,
            'label': result[0]['label'],
            'score': result[0]['score']
        })
    return sentiment_results

def plot_sentiment_distribution(sentiments):
    labels = [sentiment['label'] for sentiment in sentiments]
    counts = Counter(labels)
    
    fig = px.pie(values=list(counts.values()), names=list(counts.keys()), title="Sentiment Distribution")
    fig_html = fig.to_html(full_html=False)
    return fig_html

# def analyze_keyword_sentiment(comments, keyword):
#     keyword_comments = [comment for comment in comments if keyword.lower() in comment.lower()]
#     return analyze_sentiment(keyword_comments)

# def generate_word_cloud(comments, title):
#     text = ' '.join(comments)
#     wordcloud = WordCloud().generate(text)

#     plt.figure(figsize=(10,5))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     plt.title(title)
#     plt.show()

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    youtube_url = request.form.get('youtube_url')
    keywords = request.form.get('keywords', '').split(',')

    # Extract video ID from the YouTube URL
    video_id = youtube_url.split("v=")[1]

    # Fetch YouTube comments
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        return "Error: YouTube API key not found in environment variables", 500
    comments = fetch_youtube_comments(video_id, api_key)

    # Analyze sentiment
    sentiment_results = analyze_sentiment(comments)

    # Generate sentiment distribution pie chart
    sentiment_pie_chart = plot_sentiment_distribution(sentiment_results)

    # Render the results page with sentiment data and visualization
    return render_template('results.html', youtube_url=youtube_url, keywords=keywords, 
                           sentiment_results=sentiment_results, sentiment_pie_chart=sentiment_pie_chart)

# Running Flask with Dash
if __name__ == "__main__":
    app.run(debug=True)
