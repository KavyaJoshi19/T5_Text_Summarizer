# app.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os

# Import our T5SummarizationModel
from model import T5SummarizationModel

# Initialize FastAPI app
app = FastAPI(title="T5 Text Summarizer")

# Create templates and static directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Create templates directory for HTML templates
templates = Jinja2Templates(directory="templates")

# Initialize the T5 summarizer
# Use t5-small for faster performance or your fine-tuned model
summarizer = T5SummarizationModel("t5-small")

# HTML template for the home page
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>T5 Text Summarizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background-color: #f9f9f9;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        textarea {
            width: 100%;
            height: 200px;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 20px;
            font-family: inherit;
            font-size: 16px;
            box-sizing: border-box;
            resize: vertical;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            width: 30%;
        }
        input[type="number"] {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
            width: 100%;
        }
        button:hover {
            background-color: #2980b9;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 14px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <h1>T5 Text Summarizer</h1>
    <div class="container">
        <form action="/summarize" method="post">
            <label for="text">Enter your text to summarize:</label>
            <textarea name="text" id="text" placeholder="Paste your text here..." required></textarea>
            
            <div class="controls">
                <div class="control-group">
                    <label for="max_length">Max Length:</label>
                    <input type="number" id="max_length" name="max_length" min="30" max="500" value="150">
                </div>
                
                <div class="control-group">
                    <label for="min_length">Min Length:</label>
                    <input type="number" id="min_length" name="min_length" min="10" max="200" value="40">
                </div>
                
                <div class="control-group">
                    <label for="num_beams">Num Beams:</label>
                    <input type="number" id="num_beams" name="num_beams" min="1" max="8" value="4">
                </div>
            </div>
            
            <button type="submit">Generate Summary</button>
        </form>
    </div>
    
    <div class="footer">
        <p>Powered by Hugging Face's T5 Model</p>
    </div>
</body>
</html>
"""

# HTML template for the results page
RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>T5 Text Summarizer - Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background-color: #f9f9f9;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .summary-container {
            background-color: #e8f4fc;
            padding: 20px;
            border-radius: 6px;
            border-left: 5px solid #3498db;
            margin-bottom: 25px;
        }
        h2 {
            color: #3498db;
            margin-top: 0;
        }
        .original-text {
            max-height: 200px;
            overflow-y: auto;
            padding: 15px;
            background-color: #f1f1f1;
            border-radius: 6px;
            margin-bottom: 20px;
        }
        a.button {
            display: inline-block;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            padding: 12px 20px;
            font-size: 16px;
            border-radius: 4px;
            transition: background-color 0.3s;
            text-align: center;
        }
        a.button:hover {
            background-color: #2980b9;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 14px;
            color: #7f8c8d;
        }
        .metrics {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .metric {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
            flex: 1;
            margin: 0 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <h1>Summary Results</h1>
    
    <div class="container">
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{{original_length}}</div>
                <div class="metric-label">Original Characters</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{summary_length}}</div>
                <div class="metric-label">Summary Characters</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{compression_ratio}}%</div>
                <div class="metric-label">Compression</div>
            </div>
        </div>
        
        <h2>Generated Summary</h2>
        <div class="summary-container">
            {{summary}}
        </div>
        
        <h2>Original Text</h2>
        <div class="original-text">
            {{original_text}}
        </div>
        
        <a href="/" class="button">Back to Summarizer</a>
    </div>
    
    <div class="footer">
        <p>Powered by Hugging Face's T5 Model</p>
    </div>
</body>
</html>
"""

# Write templates to files
with open("templates/home.html", "w") as f:
    f.write(HOME_TEMPLATE)

with open("templates/results.html", "w") as f:
    f.write(RESULTS_TEMPLATE)

# Define routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/summarize", response_class=HTMLResponse)
async def generate_summary(
    request: Request,
    text: str = Form(...),
    max_length: int = Form(150),
    min_length: int = Form(40),
    num_beams: int = Form(4)
):
    # Generate summary using our model
    summary = summarizer.summarize(
        text=text,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams
    )
    
    # Calculate metrics
    original_length = len(text)
    summary_length = len(summary)
    compression_ratio = round((1 - (summary_length / original_length)) * 100, 1)
    
    return templates.TemplateResponse(
        "results.html", 
        {
            "request": request,
            "summary": summary,
            "original_text": text,
            "original_length": original_length,
            "summary_length": summary_length,
            "compression_ratio": compression_ratio
        }
    )

# Run the app
if __name__ == "__main__":
    print("Starting T5 Summarization Web UI...")
    print("Once started, open your browser to http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)