# SOCAR Process Analyst

![SOCAR Process Analyst Logo](https://via.placeholder.com/800x200.png?text=SOCAR+Process+Analyst)

## Project Overview

SOCAR Process Analyst is a comprehensive data analytics and visualization solution for oil and gas processing operations. This platform provides three different interfaces to access and leverage process data insights:

1. **GitHub Repository** - Access to source code, analysis notebooks, and development resources
2. **Web Dashboard** - Interactive visualization and analysis through a browser interface
3. **Telegram Bot** - Quick insights and alerts through convenient messaging

Each interface serves different user needs while accessing the same underlying analytics engine.

## Core Features

- **Process Efficiency Analysis**: Identify optimal operating parameters to maximize output
- **Energy Consumption Optimization**: Track and reduce energy usage across operations
- **Environmental Impact Assessment**: Monitor and minimize CO2 emissions
- **Catalyst Performance Evaluation**: Compare and optimize catalyst usage
- **Interactive Visualizations**: Filter and explore process data dynamically
- **Safety Incident Tracking**: Monitor and improve safety performance

## Data Insights

The platform analyzes process data with 23 key fields including:

- Process types and steps
- Operating parameters (temperature, pressure, duration)
- Resource metrics (energy, catalysts, worker count)
- Efficiency metrics (processing efficiency, energy per ton)
- Environmental impact (CO2 emissions)
- Economic indicators (operational costs, cost per ton)

## Three Ways to Access SOCAR Process Analyst

### 1. GitHub Repository
**URL**: [https://github.com/Ismat-Samadov/SOCAR_ProcessAnalyst](https://github.com/Ismat-Samadov/SOCAR_ProcessAnalyst)

The GitHub repository provides:

- **Source Code Access**: View and download all project components
- **Jupyter Notebooks**: Run detailed analysis scripts locally
- **Documentation**: Comprehensive project documentation
- **Development Resources**: Contribute to or extend the platform

#### Repository Structure

```
SOCAR_ProcessAnalyst/
├── README.md                      # Main project README
├── analysis/                      # Data analysis component
│   ├── README.md                  # Analysis module documentation
│   ├── analyse.ipynb              # Jupyter notebook for analysis
│   └── data/                      # Data directory
│       ├── charts/                # Generated chart images
│       ├── data.csv               # Process data (CSV format)
│       └── data.xlsx              # Process data (Excel format)
└── dashboard/                     # Interactive dashboard component
    ├── README.md                  # Dashboard documentation
    ├── app.py                     # Flask application
    ├── data/                      # Dashboard data
    │   └── data.csv               # Process data for dashboard
    ├── requirements.txt           # Python dependencies
    ├── static/                    # Static assets
    │   ├── css/                   # Stylesheets
    │   │   └── style.css          # Dashboard styling
    │   └── js/                    # JavaScript files
    │       └── main.js            # Dashboard interactivity
    └── templates/                 # HTML templates
        └── index.html             # Dashboard HTML template
```

#### Getting Started with the Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/Ismat-Samadov/SOCAR_ProcessAnalyst.git
   cd SOCAR_ProcessAnalyst
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis notebooks:
   ```bash
   cd analysis
   jupyter notebook analyse.ipynb
   ```

5. Launch the dashboard locally:
   ```bash
   cd dashboard
   python app.py
   ```

### 2. Web Dashboard
**URL**: [https://socar-processanalyst.onrender.com/](https://socar-processanalyst.onrender.com/)

The web dashboard provides:

- **Interactive Visualizations**: Explore data through dynamic charts and graphs
- **Filtering Capabilities**: Drill down into specific processes, timeframes, or parameters
- **Performance Metrics**: Track KPIs and benchmarks in real-time
- **Mobile-Responsive Design**: Access insights from any device

#### Dashboard Features

- **Overview Panel**: High-level metrics and KPIs
- **Process Analysis**: Detailed breakdowns by process type and step
- **Efficiency Tracker**: Monitor energy, environmental, and cost efficiency
- **Catalyst Comparison**: Compare performance across different catalysts
- **Custom Reports**: Generate tailored reports for specific needs

#### Using the Dashboard

1. Navigate to [https://socar-processanalyst.onrender.com/](https://socar-processanalyst.onrender.com/)
2. Use the date range selector to focus on specific time periods
3. Apply filters to analyze specific process types or equipment
4. Interact with charts to drill down into data
5. Export reports or visualizations as needed

### 3. Telegram Bot
**URL**: [https://web.telegram.org/k/#@socar_analyst_bot](https://web.telegram.org/k/#@socar_analyst_bot)

The Telegram bot provides:

- **Quick Insights**: Get key metrics and stats on demand
- **Automated Alerts**: Receive notifications about process anomalies
- **Scheduled Reports**: Get daily or weekly performance summaries
- **Convenient Access**: Use from any device with Telegram installed

#### Bot Commands

- `/start` - Begin interaction with the bot
- `/help` - View available commands and instructions
- `/summary` - Get current process performance summary
- `/efficiency` - View recent efficiency metrics
- `/alert` - Configure alert thresholds and notifications
- `/report` - Generate and receive custom reports

#### Using the Telegram Bot

1. Open Telegram and search for `@socar_analyst_bot`
2. Start a conversation with the bot
3. Use commands to request specific information
4. Configure alert preferences for automatic notifications
5. Schedule regular reports based on your needs

## Technologies Used

- **Data Analysis**: Python, Pandas, NumPy, SciPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly.js
- **Web Dashboard**: Flask, Bootstrap, HTML/CSS/JavaScript
- **Telegram Bot**: Python-telegram-bot, API integration
- **Deployment**: Render (cloud platform)

## Project Roadmap

### Current Capabilities
- Statistical analysis of process parameters
- Interactive data visualization
- Performance metric tracking
- Basic reporting and alerts

### Upcoming Features
- Machine learning models for process optimization
- Predictive maintenance algorithms
- Anomaly detection for quality control
- Real-time optimization recommendations
- Advanced integration with production systems

## Prerequisites and Requirements

- **For Repository Use**: 
  - Python 3.9+
  - Git for version control
  - Dependencies listed in requirements.txt
  
- **For Dashboard Access**:
  - Modern web browser (Chrome, Firefox, Safari, Edge)
  - Internet connection
  
- **For Telegram Bot**:
  - Telegram account
  - Mobile device or desktop Telegram client

## Team and Support

SOCAR Process Analyst is developed and maintained by the Ismat Samadov, combining expertise in process engineering, data science, and software development.

For support or inquiries:
- **Technical Issues**: File an issue on the GitHub repository
- **Dashboard Feedback**: Use the feedback form on the dashboard
- **Bot Assistance**: Use the `/help` command in Telegram


