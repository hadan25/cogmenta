import json
import os

def visualize_kg(triples, filename="knowledge_graph.html"):
    """
    Create a simple visualization of the knowledge graph using HTML and JavaScript
    """
    # Filter to remove low confidence triples
    filtered_triples = [t for t in triples if t['confidence'] > 0.3]
    
    # Extract unique nodes
    nodes = set()
    for triple in filtered_triples:
        nodes.add(triple['subject'])
        nodes.add(triple['object'])
    
    # Create node list for visualization
    node_list = [{"id": node, "label": node.replace('_', ' ').title()} for node in nodes]
    
    # Create edge list for visualization
    edge_list = []
    for i, triple in enumerate(filtered_triples):
        edge = {
            "id": f"e{i}",
            "source": triple['subject'],
            "target": triple['object'],
            "label": triple['predicate'].replace('_', ' '),
            "weight": triple['confidence']
        }
        edge_list.append(edge)
    
    # Create the HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knowledge Graph Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/vis-network.min.js"></script>
        <style>
            #mynetwork {
                width: 100%;
                height: 800px;
                border: 1px solid lightgray;
            }
            .legend {
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.8);
                border: 1px solid #ccc;
                position: absolute;
                bottom: 10px;
                right: 10px;
                z-index: 10;
            }
        </style>
    </head>
    <body>
        <h1>Cogmenta Core - Knowledge Graph</h1>
        <div id="mynetwork"></div>
        <div class="legend">
            <h3>Confidence Levels</h3>
            <div><span style="color: green; font-weight: bold;">━━━</span> High (0.8-1.0)</div>
            <div><span style="color: blue; font-weight: bold;">━━━</span> Medium (0.5-0.79)</div>
            <div><span style="color: red; font-weight: bold;">━━━</span> Low (0.3-0.49)</div>
        </div>
        
        <script>
            // Create nodes and edges
            const nodes = GRAPH_NODES;
            const edges = GRAPH_EDGES;
            
            // Configure options
            const options = {
                nodes: {
                    shape: 'circle',
                    size: 25,
                    font: {
                        size: 16
                    }
                },
                edges: {
                    arrows: 'to',
                    smooth: {
                        type: 'curvedCW',
                        roundness: 0.2
                    },
                    font: {
                        size: 12,
                        align: 'middle'
                    }
                },
                physics: {
                    stabilization: true,
                    barnesHut: {
                        gravitationalConstant: -5000,
                        springConstant: 0.001,
                        springLength: 200
                    }
                },
                layout: {
                    improvedLayout: true
                }
            };
            
            // Create network
            const container = document.getElementById('mynetwork');
            
            // Color edges based on confidence
            edges.forEach(edge => {
                if (edge.weight >= 0.8) {
                    edge.color = {color: 'green', highlight: 'green'};
                    edge.width = 3;
                } else if (edge.weight >= 0.5) {
                    edge.color = {color: 'blue', highlight: 'blue'};
                    edge.width = 2;
                } else {
                    edge.color = {color: 'red', highlight: 'red'};
                    edge.width = 1;
                }
            });
            
            const data = {
                nodes: new vis.DataSet(nodes),
                edges: new vis.DataSet(edges)
            };
            
            const network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """.replace('GRAPH_NODES', json.dumps(node_list)).replace('GRAPH_EDGES', json.dumps(edge_list))
    
    # Write to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        return os.path.abspath(filename)
    except UnicodeEncodeError:
        # Fallback for systems with limited encoding support
        html_ascii = html.encode('ascii', 'xmlcharrefreplace').decode('ascii')
        with open(filename, 'w', encoding='ascii') as f:
            f.write(html_ascii)
        return os.path.abspath(filename)