# cogmenta_core/visualization/reasoning_trace_viz.py
import json
import os
import time

def visualize_reasoning(reasoning_traces, filename="reasoning_traces.html"):
    """
    Create a visualization of reasoning traces
    """
    if not reasoning_traces:
        print("No reasoning traces to visualize")
        return None
    
    # Format timestamps for display
    for trace in reasoning_traces:
        if 'timestamp' in trace:
            trace['time'] = time.strftime('%H:%M:%S', time.localtime(trace['timestamp']))
    
    # Create the HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reasoning Trace Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            .reasoning-trace {
                border: 1px solid #ccc;
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .strategy {
                font-weight: bold;
                color: white;
                padding: 3px 8px;
                border-radius: 3px;
                margin-left: 10px;
            }
            .symbolic { background-color: #2c3e50; }
            .neural { background-color: #8e44ad; }
            .abductive { background-color: #d35400; }
            .hybrid { background-color: #16a085; }
            
            .success-score {
                float: right;
                font-weight: bold;
            }
            .high { color: green; }
            .medium { color: blue; }
            .low { color: red; }
            
            .features {
                margin-top: 10px;
                font-style: italic;
                color: #666;
            }
            
            .scores {
                margin-top: 10px;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
            
            .score-bar {
                height: 20px;
                margin: 5px 0;
                background-color: #eee;
                position: relative;
            }
            
            .score-fill {
                height: 100%;
                position: absolute;
                left: 0;
                top: 0;
            }
            
            .symbolic-score { background-color: #2c3e50; }
            .neural-score { background-color: #8e44ad; }
            .abductive-score { background-color: #d35400; }
            .hybrid-score { background-color: #16a085; }
            
            .score-label {
                position: absolute;
                left: 10px;
                top: 0;
                color: white;
                font-weight: bold;
                line-height: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Cogmenta Core - Reasoning Traces</h1>
        <p>Showing the most recent reasoning traces and decision making process.</p>
        
        <div id="traces">
            TRACE_CONTENT
        </div>
        
        <script>
            // Parse the reasoning traces
            const traces = TRACE_DATA;
            let traceHTML = '';
            
            traces.forEach(trace => {
                // Determine success score class
                let scoreClass = 'medium';
                if (trace.success_score >= 0.7) scoreClass = 'high';
                if (trace.success_score < 0.4) scoreClass = 'low';
                
                // Create trace HTML
                let html = `
                    <div class="reasoning-trace">
                        <h3>
                            ${trace.time}: "${trace.query}"
                            <span class="strategy ${trace.selected_strategy}">${trace.selected_strategy}</span>
                            ${trace.success_score !== undefined ? 
                              `<span class="success-score ${scoreClass}">Score: ${trace.success_score.toFixed(2)}</span>` : ''}
                        </h3>
                        
                        <div class="features">
                            Features: ${Object.entries(trace.features)
                                           .map(([key, value]) => `${key}=${value}`)
                                           .join(', ')}
                        </div>
                        
                        <div class="scores">
                            <h4>Strategy Scores:</h4>
                            ${Object.entries(trace.scores).map(([strategy, score]) => `
                                <div>
                                    <div class="score-bar">
                                        <div class="score-fill ${strategy}-score" style="width: ${score * 100}%">
                                            <div class="score-label">${strategy}: ${score.toFixed(2)}</div>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
                
                traceHTML = html + traceHTML;  // Prepend to show newest first
            });
            
            document.getElementById('traces').innerHTML = traceHTML;
        </script>
    </body>
    </html>
    """.replace('TRACE_DATA', json.dumps(reasoning_traces)).replace('TRACE_CONTENT', '')
    
    # Write to file
    with open(filename, 'w') as f:
        f.write(html)
    
    print(f"Reasoning trace visualization saved to {os.path.abspath(filename)}")
    return os.path.abspath(filename)