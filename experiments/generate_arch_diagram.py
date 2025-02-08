import graphviz

def create_microservices_architecture_diagram():
    # Create a new directed graph
    dot = graphviz.Digraph('microservices_architecture', comment='Microservices Architecture',
                            node_attr={
                                'style': 'filled',
                                'fontname': 'Arial',
                                'fontsize': '10'
                            },
                            edge_attr={
                                'fontname': 'Arial',
                                'fontsize': '8'
                            })
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='8,5')
    dot.attr('node', shape='box')

    # Define color palette
    colors = {
        'database': '#6495ED',  # Cornflower Blue
        'service': '#90EE90',   # Light Green
        'message_broker': '#FFA07A',  # Light Salmon
        'api_gateway': '#DDA0DD'  # Plum
    }
    # Set node positions using subgraphs for ranks
    with dot.subgraph() as s:
        s.attr(rank='max')
        s.node('mongodb')

    with dot.subgraph() as s:
        s.attr(rank='min')
        s.node('kong')
    # Add nodes
    # Services
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('order_service')
        s.node('user_service_v1')
        s.node('user_service_v2')
    dot.node('order_service', 'Order Service\nPort: 5001', 
             fillcolor=colors['service'], 
             color='darkgreen')
    dot.node('user_service_v1', 'User Service V1\nPort: 5002', 
             fillcolor=colors['service'], 
             color='darkgreen')
    dot.node('user_service_v2', 'User Service V2\nPort: 5003', 
             fillcolor=colors['service'], 
             color='darkgreen')

    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('rabbitmq', 'RabbitMQ\nMessage Broker\nPorts: 5673, 15672', 
                fillcolor=colors['message_broker'], 
                color='darkorange',
                shape='oval')

    # API Gateway
    dot.node('kong', 'Kong API Gateway\nPorts: 8000, 8001', 
             fillcolor=colors['api_gateway'], 
             color='darkviolet',
             shape='rect',
             width='4')

    # Databases
    dot.node('mongodb', 'MongoDB\nDatabase', 
             fillcolor=colors['database'], 
             color='darkblue',
             shape='cylinder')


    # Add edges (connections)
    # Database Connections
    dot.edge('mongodb', 'order_service', label='Data Storage', style='dashed', dir='both')
    dot.edge('mongodb', 'user_service_v1', label='Data Storage', style='dashed', dir='both')
    dot.edge('mongodb', 'user_service_v2', label='Data Storage', style='dashed', dir='both')

    # Message Queue Connections
    dot.edge( 'rabbitmq', 'order_service', label='Subscribe', color='darkorange')
    dot.edge('user_service_v1', 'rabbitmq', label='Publish', color='darkorange')
    dot.edge('user_service_v2', 'rabbitmq', label='Publish', color='darkorange')

    # API Gateway Routing
    dot.edge('kong', 'order_service', label='Route Requests', style='dotted')
    dot.edge('kong', 'user_service_v1', label='Route Requests', style='dotted')
    dot.edge('kong', 'user_service_v2', label='Route Requests', style='dotted')

    # Render the graph
    dot.render('microservices_architecture', format='png', cleanup=True)
    dot.render('microservices_architecture', format='svg', cleanup=True)
    print("Architecture diagram generated as microservices_architecture.png and microservices_architecture.svg")

if __name__ == '__main__':
    create_microservices_architecture_diagram()