from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import cloudpickle as pickle
from issue_model import IssueModel
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
import cloudpickle as pickle
from io import BytesIO
import os, base64, tempfile, requests
import matplotlib.pyplot as plt
import seaborn as sns
import re  

app = Flask(__name__)

# Load trained model
with open('search_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Store the last matched row for follow-up context
last_context_row = None

# Dummy function for LLM call (replace with actual API later)
def call_llm(prompt, context=None):
    if context:
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        return f"[LLM Response Placeholder]\n\nPrompt: {prompt}\n\nContext:\n{context_str}"
    return f"[LLM Suggestion Placeholder] Based on your issue: '{prompt}', please check system logs or network connectivity."

@app.route('/')
def index():
    return render_template('index.html')



# @app.route('/search', methods=['POST'])
# def search():
    # global last_context_row
    # query = request.json.get('query').lower()

    # --- Step 1: Intent Detection ---
    # def is_analysis_query(q):
        # keywords = ['total', 'count', 'how many', 'number of', 'yesterday', 'today', 'last week']
        # return any(k in q for k in keywords)

    # --- Step 2: Handle Analysis Queries ---
    # if is_analysis_query(query):
        # df = model.df.copy()
        # df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
        # df = df.dropna(subset=['Date'])

        # today = pd.Timestamp.today().normalize()
        # yesterday = today - pd.Timedelta(days=1)

        # query = query.lower()  # Normalize the case for easier matching

        # YESTERDAY queries
        # if 'yesterday' in query and (
            # 'count' in query or
            # 'total' in query or
            # 'issues' in query or
            # 'resolved' in query or
            # 'occurred' in query or
            # 'number' in query
        # ):
            # count = df[df['Date'] == yesterday].shape[0]
            # return jsonify({
                # 'analysis': True,
                # 'query_type': 'count_yesterday',
                # 'date': str(yesterday.date()),
                # 'total_issues': count
            # })

        # elif 'today' in query and (
        # 'count' in query or
        # 'total' in query or
        # 'issues' in query or
        # 'resolved' in query or
        # 'occurred' in query or
        # 'number' in query
        # ):
            # count = df[df['Date'] == today].shape[0]
            # return jsonify({
                # 'analysis': True,
                # 'query_type': 'count_today',
                # 'date': str(today.date()),
                # 'total_issues': count
            # })

        # elif (
            # 'total' in query or
            # 'count' in query or
            # 'all' in query or
            # 'overall' in query or
            # ('issues' in query and ('how many' in query or 'number' in query))
        # ):
            # total = df.shape[0]
            # return jsonify({
                # 'analysis': True,
                # 'query_type': 'total_issues',
                # 'total_issues': total
            # })


    # --- Step 3: Proceed with Normal RCA Flow ---
    # query_vec = model.vectorizer.transform([query])
    # sim_scores = cosine_similarity(query_vec, model.tfidf_matrix).flatten()

    # top_indices = sim_scores.argsort()[-5:][::-1]
    # top_scores = sim_scores[top_indices]

    # top_rows = model.df.iloc[top_indices].copy()
    # top_rows['similarity'] = top_scores
    # top_rows['Date'] = pd.to_datetime(top_rows['Date'], errors='coerce')
    # top_rows = top_rows.dropna(subset=['Date'])
    # top_rows = top_rows.sort_values(by='Date', ascending=True)

    # results = []
    # for i, (_, row) in enumerate(top_rows.iterrows(), start=1):
        # last_context_row = row.to_dict()
        # formatted_date = row['Date'].strftime('%d-%m-%y')
        # results.append({
            # 'Index': i,
            # 'TrueIndex': int(row['Index']),
            # 'Date': formatted_date,
            # 'INC': row.get('INC', ''),
            # 'INC Priority': row.get('INC Priority', ''),
            # 'Description': row.get('Description', ''),
            # 'Resolved By': row.get('Resolved By', ''),
            # 'Action': row.get('Action', 'N/A'),
            # 'RCA': row.get('RCA', 'N/A'),
            # 'Count': row.get('Count', 1)
        # })

    # if not results:
        # ai_response = call_llm(query)
        # return jsonify({
            # 'ai_generated': True,
            # 'disclaimer': "AI-generated response. May differ from actual resolution.",
            # 'response': ai_response
        # })

    # return jsonify(results)
    
    
def safe_format_date(date_obj, fmt='%d-%m-%y'):
    try:
        return pd.to_datetime(date_obj, errors='coerce').strftime(fmt)
    except:
        return 'Unknown'


from flask import request, jsonify
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

last_context_row = None  # Global variable to store last matched row

@app.route('/search', methods=['POST'])
def search():
    global last_context_row
    query = request.json.get('query', '').lower()
    df = model.df.copy()

    # --- Step 1: Intent Detection ---
    def is_analysis_query(q):
        keywords = ['total', 'count', 'how many', 'number of', 'yesterday', 'today', 'last week']
        return any(k in q for k in keywords)
    def extract_specific_date(query, exclude_dates=[]):
        try:
            parsed = parser.parse(query, fuzzy=True).date()
            normalized = pd.Timestamp(parsed).normalize()
            if normalized not in exclude_dates:
                return normalized
        except:
            return None

    # --- Step 2: Handle Analysis Queries ---
    if is_analysis_query(query):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
        df = df.dropna(subset=['Date'])

        today = pd.Timestamp.today().normalize()
        yesterday = today - pd.Timedelta(days=1)
        # specific_date = extract_specific_date(query, exclude_dates=[today, yesterday])


        if 'yesterday' in query:
            count = df[df['Date'] == yesterday].shape[0]
            return jsonify({'analysis': True, 'query_type': 'count_yesterday', 'date': str(yesterday.date()), 'total_issues': count})

        elif 'today' in query:
            count = df[df['Date'] == today].shape[0]
            return jsonify({'analysis': True, 'query_type': 'count_today', 'date': str(today.date()), 'total_issues': count})
         # ✅ Specific date (e.g. "How many issues on July 15?")
        
        # elif specific_date is not None:
            # count = df[df['Date'] == specific_date].shape[0]
            # return jsonify({
                # 'analysis': True,
                # 'query_type': 'count_by_specific_date',
                # 'date': str(specific_date.date()),
                # 'total_issues': count
            # })

        elif any(k in query for k in ['total', 'count', 'all', 'overall']) or ('issues' in query and ('how many' in query or 'number' in query)):
            total = df.shape[0]
            return jsonify({'analysis': True, 'query_type': 'total_issues', 'total_issues': total})

    # --- Step 3: Handle Incident-Specific Queries ---
    incident_match = re.search(r'inc\d{5,}', query)
    if incident_match:
        incident_id = incident_match.group(0).upper()
        incident_row = df[df['INC'].str.upper() == incident_id]

        if not incident_row.empty:
            row = incident_row.iloc[0]
            last_context_row = row.to_dict()
            return jsonify({
                'incident_match': True,
                'INC': row.get('INC', ''),
                # 'Date': row['Date'].strftime('%d-%m-%y'),
                'Date': safe_format_date(row['Date']),
                'INC Priority': row.get('INC Priority', ''),
                'Market': row.get('Market', ''),
                'Description': row.get('Description', ''),
                'Resolved By': row.get('Resolved By', ''),
                'Action': row.get('Action', 'N/A'),
                'RCA': row.get('RCA', 'N/A'),
                'Count': row.get('Count', 1)
            })
    

    # --- Step 4: Semantic RCA Flow (TF-IDF search) ---
    query_vec = model.vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, model.tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-5:][::-1]
    top_scores = sim_scores[top_indices]

    top_rows = df.iloc[top_indices].copy()
    top_rows['similarity'] = top_scores
    top_rows['Date'] = pd.to_datetime(top_rows['Date'], errors='coerce')
    top_rows = top_rows.dropna(subset=['Date'])
    top_rows = top_rows.sort_values(by='Date', ascending=True)

    results = []
    for i, (_, row) in enumerate(top_rows.iterrows(), start=1):
        last_context_row = row.to_dict()
        # formatted_date = row['Date'].strftime('%d-%m-%y')
        formatted_date = safe_format_date(row['Date'])
        results.append({
            'Index': i,
            'TrueIndex': int(row['Index']),
            'Date': formatted_date,
            'INC': row.get('INC', ''),
            'INC Priority': row.get('INC Priority', ''),
            'Market': row.get('Market', ''),
            'Description': row.get('Description', ''),
            'Resolved By': row.get('Resolved By', ''),
            'Action': row.get('Action', 'N/A'),
            'RCA': row.get('RCA', 'N/A'),
            'Count': row.get('Count', 1)
        })

    if results:
        return jsonify(results)

    # --- Step 5: Fallback to Follow-Up Questions Using Last Context Row ---
    if last_context_row:
        # formatted_date = pd.to_datetime(last_context_row.get('Date', ''), errors='coerce')
        formatted_date = safe_format_date(last_context_row.get('Date', ''), '%d-%m-%Y')
        # formatted_date = formatted_date.strftime('%d-%m-%Y') if not pd.isna(formatted_date) else 'Unknown'
        

        if 'rca' in query:
            return jsonify({'response': f"The RCA is: {last_context_row.get('RCA', 'Not available')}."})
        elif 'resolved by' in query or 'who fixed' in query:
            return jsonify({'response': f"This was resolved by: {last_context_row.get('Resolved By', 'Not available')}."})
        elif 'action' in query:
            return jsonify({'response': f"The action taken was: {last_context_row.get('Action', 'Not available')}."})
        elif 'date' in query:
            return jsonify({'response': f"The incident occurred on: {formatted_date}."})
        elif 'description' in query:
            return jsonify({'response': f"The description is: {last_context_row.get('Description', 'Not available')}."})
        else:
            return jsonify({
                'response': (
                    f"Here are the details I found: "
                    f"Date: {formatted_date}, Description: {last_context_row.get('Description', 'N/A')}, "
                    f"Resolved By: {last_context_row.get('Resolved By', 'N/A')}, "
                    f"Action: {last_context_row.get('Action', 'N/A')}, RCA: {last_context_row.get('RCA', 'N/A')}."
                )
            })

    # --- Step 6: Fallback to AI if No Match ---
    ai_response = call_llm(query)
    return jsonify({
        'ai_generated': True,
        'disclaimer': "AI-generated response. May differ from actual resolution.",
        'response': ai_response
    })




def safe_get(value, default='N/A'):
    # Convert NaN or non-serializable types to safe values
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return str(value)  # Convert numpy types or others to string



@app.route('/get_rca', methods=['POST'])
def get_rca():
    try:
        idx = request.json.get('index')
        if idx is None:
            return jsonify({'error': 'No index provided'}), 400

        df = model.df
        if df is None or 'Index' not in df.columns:
            return jsonify({'error': 'Data not loaded or "Index" column missing'}), 500

        matched = df[df['Index'] == int(idx)]

        if matched.empty:
            return jsonify({'error': f'No record found for index {idx}'}), 404

        row = matched.squeeze()

        def safe_get(value, default='N/A'):
            import math
            if value is None:
                return default
            if isinstance(value, float) and math.isnan(value):
                return default
            return str(value)

        return jsonify({
            'INC': safe_get(row.get('INC')),
            'INC Priority': safe_get(row.get('INC Priority')),
            'Action': safe_get(row.get('Action')),
            'RCA': safe_get(row.get('RCA')),
            'Resolved By': safe_get(row.get('Resolved By')),
            'Business Impact': safe_get(row.get('Business Impact')),
            'Month': safe_get(row.get('Month')),
            'EndTime': safe_get(row.get('EndTime')),
            'Date': safe_get(row.get('Date')),
            'Area': safe_get(row.get('Area')),
            'App': safe_get(row.get('App')),
            'Description': safe_get(row.get('Description')),
            'Caused By': safe_get(row.get('Caused By')),
            'StartTime': safe_get(row.get('StartTime')),
            'Long Running': safe_get(row.get('Long Running')),
            'Identified by': safe_get(row.get('Identified by'))
        })

    except Exception as e:
        # Log the error and return error response
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
    
@app.route('/followup', methods=['POST'])
def followup():
    global last_context_row
    followup_query = request.json.get('followup')

    if not last_context_row:
        return jsonify({'error': 'No context available for follow-up. Please search first.'}), 400

    query_lower = followup_query.lower()

    #Special intents
    if any(kw in query_lower for kw in ['who resolved', 'resolved by', 'resolved person']):
        val = last_context_row.get('Resolved By')
        if val:
            response_text = f"Resolved By: {val}"
        else:
            response_text = "Sorry, the resolver information is not available in the records."
        return jsonify({'response': response_text})

    elif any(kw in query_lower for kw in ['when it got resolved', 'resolved date', 'resolution date', 'when resolved', 'end time']):
        val = last_context_row.get('EndTime')
        if val:
            response_text = f"Resolved Date: {val}"
        else:
            response_text = "Sorry, the resolved date is not available in the records."
        return jsonify({'response': response_text})

    elif 'what was inc' in query_lower or 'incident number' in query_lower:
        inc_val = last_context_row.get('INC')
        if inc_val:
            response_text = f"The incident number is: {inc_val}"
        else:
            response_text = "Sorry, the incident number is not available. Could you please clarify?"
        return jsonify({'response': response_text})

    elif 'priority' in query_lower:
        priority_val = last_context_row.get('INC Priority')
        if priority_val:
            response_text = f"The priority of this incident is: {priority_val}"
        else:
            response_text = "Sorry, the priority information is not available in the records."
        return jsonify({'response': response_text})

    # Fallback to generic keyword matching
    field_map = {
        'month': 'Month',
        'date': 'Date',
        'resolved': 'Resolved By',
        'inc': 'INC',
        'priority': 'INC Priority',
        'description': 'Description',
        'issue details': 'INC/Issue Details',
        'business impact': 'Business Impact',
        'action': 'Action',
        'caused by': 'Caused By',
        'rca': 'RCA',
        'next steps': 'Next Steps / Actions',
        'start time': 'StartTime',
        'end time': 'EndTime',
        'long running': 'Long Running',
        'comments': 'Comments',
        'identified by': 'Identified by'
    }

    matched_field = None
    for keyword, field in field_map.items():
        if keyword in query_lower:
            matched_field = field
            break

    if matched_field:
        val = last_context_row.get(matched_field)
        if val:
            response_text = f"{matched_field}: {val}"
        else:
            response_text = f"Sorry, the {matched_field.lower()} is not available in the records."
    else:
        response_text = (
            "Sorry, I didn't quite get that. "
            "You can ask about priority, action, RCA, resolved by, or say 'new' to start over."
        )

    return jsonify({
        'ai_generated': True,
        'disclaimer': "AI-generated follow-up response. May differ from actual resolution.",
        'response': response_text
    })
    
@app.route('/incident_details', methods=['POST'])
def incident_details():
    idx = request.json.get('index')
    if idx is None:
        return jsonify({"error": "No incident index provided"}), 400
    try:
        row = model.df.loc[model.df['Index'] == int(idx)].squeeze()
        if row.empty:
            return jsonify({"error": "Incident not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    #Convert the full row to dict and return all columns
    return jsonify(row.to_dict())
    
@app.route('/analysis', methods=['GET'])
def analysis():
    app_counts = model.df['App'].value_counts().head(10)
    df_counts = app_counts.reset_index()
    df_counts.columns = ['App', 'Count']
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(data=df_counts, y='App', x='Count', palette='viridis')
    ax.set_title('Top Frequent Issues by App')
    ax.set_xlabel('Count')
    ax.set_ylabel('App')
    for i in ax.containers:
        ax.bar_label(i, fmt='%d', label_type='edge')
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    image_uri = f"data:image/png;base64,{image_base64}"
    return jsonify({'chart_html': f"<img src='{image_uri}' width='600'/>"})

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/analyze_upload', methods=['POST'])
def upload_csv():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding='latin1')

    app_counts = df['App'].value_counts().reset_index()
    app_counts.columns = ['App', 'Count']

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    app_counts.to_csv(tmpfile.name, index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=app_counts.head(10), y='App', x='Count', palette='viridis')
    plt.title('Top Frequent Issues by App')
    plt.xlabel('Count')
    plt.ylabel('App')

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    image_uri = f"data:image/png;base64,{image_base64}"

    download_link = f"/download_temp_csv?path={tmpfile.name}"
    return jsonify({
        'chart_html': f"<img src='{image_uri}' width='600'/>",
        'download_link': download_link
    })

@app.route('/download_temp_csv')
def download_temp_csv():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return "File not found.", 404
    return send_file(path, as_attachment=True)
  
if __name__ == '__main__':
    app.run(debug=True)
