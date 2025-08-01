# Smart Project Allocation System 🧠

An AI-powered web application that intelligently matches employees to projects based on skills, experience, and availability. Perfect for internship projects and demonstrating HR tech concepts!

## Features ✨

- **AI-Powered Matching**: Advanced algorithm that scores employee-project fit based on multiple factors
- **Employee Management**: Add and manage employee profiles with skills and availability
- **Project Creation**: Create projects with specific skill requirements and team size needs
- **Interactive Dashboard**: Real-time statistics and project overview
- **Smart Scheduling**: Visual project timeline and resource allocation
- **Responsive Design**: Modern, mobile-friendly interface with Bootstrap

## Technology Stack 🛠️

- **Backend**: Python Flask
- **Database**: SQLite (for simplicity)
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **AI Matching**: Custom algorithm with skill matching and scoring
- **Claude AI Integration**: Optional real AI-powered employee-project matching

## Quick Start 🚀

### Prerequisites
- Python 3.7+ installed on your system
- Basic knowledge of Python and web development

### Installation

1. **Clone or download the project files**
   ```bash
   mkdir smart-project-allocation
   cd smart-project-allocation
   ```

2. **Create the main application file**
   - Save the main Python code as `app.py`

3. **Create the templates directory structure**
   ```bash
   mkdir templates
   ```
   - Save all HTML templates in the `templates/` folder

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Claude AI (Optional)**
   ```bash
   # For real AI-powered matching, set your Anthropic API key:
   export ANTHROPIC_API_KEY=your_api_key_here
   # Without this, the system uses algorithm-based matching
   ```

6. **Test AI Integration (Optional)**
   ```bash
   python test_claude_ai.py
   ```

7. **Run the application**
   ```bash
   python app.py
   ```

8. **Open your browser**
   - Navigate to `http://localhost:5000`
   - The application will automatically create sample data on first run

## Project Structure 📁

```
smart-project-allocation/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies (includes anthropic)
├── test_claude_ai.py      # Test script for AI integration
├── README.md             # This file
├── project_allocation.db  # SQLite database (created automatically)
│
└── templates/
    ├── base.html          # Base template with navigation
    ├── index.html         # Dashboard page
    ├── employees.html     # Employee directory
    ├── add_employee.html  # Add new employee form
    ├── projects.html      # Project management
    ├── add_project.html   # Create new project form
    ├── project_matches.html # AI matching results
    └── schedule.html      # Project timeline view
```

## How It Works 🤖

### AI Matching System

The application offers two intelligent matching approaches:

#### 1. Claude AI-Powered Matching (When API key is provided)
- **Real AI Analysis**: Uses Anthropic's Claude AI to analyze employee-project compatibility
- **Natural Language Processing**: Understands project descriptions and employee skills contextually
- **Intelligent Reasoning**: Provides detailed explanations for match recommendations
- **Advanced Insights**: Identifies potential concerns and growth opportunities
- **Dynamic Scoring**: Adapts scoring based on project complexity and requirements

#### 2. Algorithm-Based Matching (Fallback/Default)

The system uses a sophisticated scoring algorithm that considers:

1. **Skill Matching** (Weight: 10 points)
   - Exact skill matches with proficiency scoring
   - Related skill detection using keyword groups

2. **Experience Bonus** (Weight: 5 points)
   - Bonus points for senior developers (5+ years)
   - Partial bonus for mid-level (2+ years)

3. **Availability Scoring** (Weight: 8 points)
   - Full bonus for available employees
   - Partial bonus for partially available
   - Penalty for busy employees

4. **Workload Balancing** (Weight: -3 points penalty)
   - Penalizes employees already on multiple projects
   - Promotes fair work distribution

### Sample Data

The application comes with 8 sample employees across different departments:
- **Engineering**: Alice (Python/Django), Bob (JavaScript/React), Grace (Java/Spring)
- **Design**: Carol (UI/UX/Figma)
- **DevOps**: David (AWS/Docker/Kubernetes)
- **Data Science**: Emma (Python/ML/TensorFlow)
- **Management**: Frank (Project Management/Scrum)
- **QA**: Henry (Test Automation/Selenium)

And 2 sample projects:
- **E-commerce Mobile App**: Requires React, JavaScript, UI/UX, ML
- **Data Analytics Dashboard**: Requires Python, SQL, ML, UI/UX

## Key Features Demo 🎯

### 1. Employee Management
- Add employees with skills and proficiency levels
- Track availability and current workload
- Department-based organization

### 2. AI Team Matching
- Input project requirements and get ranked employee suggestions
- Visual match percentage and skill overlap
- One-click team selection and assignment

### 3. Project Scheduling
- Visual timeline of all active projects
- Team allocation tracking
- Resource utilization overview

### 4. Dashboard Analytics
- Real-time statistics on employees and projects
- Quick action buttons for common tasks
- Recent project overview

## Usage Examples 💡

### Adding an Employee
```
Name: John Smith
Email: john.smith@company.com
Department: Engineering
Skills: Python:9, React:7, SQL:8, Docker:6
Experience: 4 years
Availability: Available
```

### Creating a Project
```
Project Name: Customer Portal Redesign
Description: Modernize the customer portal with new UI/UX and backend improvements
Required Skills: React:High, Python:Medium, UI/UX Design:High
Priority: High
Team Size: 3
Timeline: 2 months
```

### AI Matching Results
The system will automatically:
1. Analyze all employees against project requirements
2. Score each employee based on skill match, experience, and availability
3. Rank candidates by percentage match
4. Highlight exact skill matches and related expertise
5. Allow one-click team selection

## Customization Ideas 💡

To make this project your own, consider adding:

- **Enhanced AI**: Machine learning models for better matching
- **Notifications**: Email alerts for project assignments
- **Calendar Integration**: Sync with Google Calendar or Outlook
- **Reporting**: Generate PDF reports of team allocations
- **User Authentication**: Login system with role-based access
- **API Endpoints**: REST API for mobile app integration
- **Advanced Scheduling**: Gantt charts and dependency tracking
- **Performance Metrics**: Track project success rates and team performance
- **Skills Development**: Suggest training based on project needs
- **Resource Forecasting**: Predict future staffing needs

## Troubleshooting 🔧

### Common Issues

1. **Database errors**: Delete `project_allocation.db` and restart the app
2. **Port conflicts**: Change the port in `app.run(port=5001)`
3. **Template not found**: Ensure all HTML files are in the `templates/` folder
4. **Styling issues**: Check that Bootstrap CDN links are working
5. **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`

### Performance Tips

- For production: Use PostgreSQL instead of SQLite
- Add database indexing for large employee datasets
- Implement caching for frequently accessed data
- Use environment variables for configuration

## Learning Outcomes 📚

This project demonstrates:

- **Full-stack web development** with Flask
- **Database design** and ORM usage with SQLAlchemy
- **Algorithm development** for matching systems
- **UI/UX design** with modern web standards
- **Project structure** and code organization
- **API design** and JSON handling
- **Responsive web design** with Bootstrap
- **Interactive JavaScript** functionality

## API Endpoints 🔌

The application includes several API endpoints:

- `GET /api/employee/<id>` - Get employee details
- `POST /projects/<id>/assign` - Assign employees to project
- `GET /projects/<id>/match` - Get AI matching results

## Security Considerations 🔒

For production deployment:
- Add user authentication and authorization
- Implement CSRF protection
- Use environment variables for secrets
- Add input validation and sanitization
- Enable HTTPS
- Implement rate limiting

## Future Enhancements 🚀

**Phase 2 Features:**
- User authentication system
- Role-based access control (Admin, Manager, Employee)
- Email notifications for assignments
- Calendar integration
- Advanced reporting and analytics

**Phase 3 Features:**
- Machine learning for improved matching
- Mobile application
- Integration with HR systems
- Performance tracking and metrics
- Skill gap analysis

## Contributing 🤝

This project is perfect for learning and experimentation. Feel free to:
- Add new features and improvements
- Enhance the AI matching algorithm
- Improve the user interface
- Add more comprehensive testing
- Optimize database queries
- Create additional integrations

## Development Setup 👩‍💻

For development:

1. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install in development mode**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run with debug mode**:
   ```bash
   export FLASK_DEBUG=1  # On Windows: set FLASK_DEBUG=1
   python app.py
   ```

## Testing 🧪

To test the application:

1. **Manual Testing**:
   - Test employee creation with various skill combinations
   - Create projects with different requirements
   - Verify AI matching results make sense
   - Test team assignment workflow

2. **Edge Cases**:
   - Empty database scenarios
   - Projects with no matching employees
   - Employees with no skills
   - Invalid date ranges

## Deployment 🌐

For deploying to production:

1. **Heroku**:
   ```bash
   # Create Procfile
   echo "web: python app.py" > Procfile
   
   # Deploy
   git init
   git add .
   git commit -m "Initial commit"
   heroku create your-app-name
   git push heroku main
   ```

2. **VPS/Cloud Server**:
   - Use gunicorn as WSGI server
   - Set up nginx as reverse proxy
   - Use PostgreSQL for database
   - Implement SSL certificates

## License 📄

This project is created for educational purposes. Feel free to use and modify for your learning and internship projects.

## Support 💬

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify file structure matches the documentation
4. Check that all HTML templates are in the `templates/` folder

---

**Perfect for**: Internships, Portfolio Projects, HR Tech Demos, Learning Flask, AI Algorithm Practice

**Estimated Development Time**: 2-3 days for basic version, 1-2 weeks for full features

**Difficulty Level**: Intermediate (good for showing full-stack skills)

Good luck with your internship project! 🚀

## Screenshots 📸

The application features:
- **Modern Dashboard** with statistics cards and quick actions
- **Employee Directory** with skill badges and availability status
- **AI Matching Interface** with percentage scores and visual indicators
- **Project Management** with priority levels and timeline tracking
- **Interactive Forms** with validation and user-friendly inputs
- **Responsive Design** that works perfectly on desktop and mobile

*Note: The application automatically generates sample data on first run, so you can immediately see all features in action!*