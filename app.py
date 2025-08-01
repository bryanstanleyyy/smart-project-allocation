# Smart Project Allocation System
# A Flask web application for AI-powered employee-project matching

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import json
import re
from collections import Counter
import os
from werkzeug.exceptions import BadRequest
from sqlalchemy import and_
import logging
import google.generativeai as genai
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///project_allocation.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models
class Employee(db.Model):
    __tablename__ = 'employees'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    department = db.Column(db.String(50), nullable=False, index=True)
    skills = db.Column(db.Text, nullable=False)  # JSON string of skills with proficiency
    experience_years = db.Column(db.Integer, nullable=False, default=0)
    availability = db.Column(db.String(20), default='Available', index=True)  # Available, Busy, Partially Available
    current_projects = db.Column(db.Text, default='[]')  # JSON list of project IDs
    hourly_rate = db.Column(db.Float, default=0.0)
    phone = db.Column(db.String(20))
    hire_date = db.Column(db.Date, default=lambda: datetime.utcnow().date())
    is_active = db.Column(db.Boolean, default=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    assignments = db.relationship('ProjectAssignment', backref='employee', lazy='dynamic', cascade='all, delete-orphan')

    def get_skills_list(self):
        try:
            return json.loads(self.skills) if self.skills else []
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Invalid JSON in skills for employee {self.id}")
            return []
    
    def get_current_projects(self):
        try:
            return json.loads(self.current_projects) if self.current_projects else []
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Invalid JSON in current_projects for employee {self.id}")
            return []
    
    def get_active_assignments(self):
        """Get active project assignments for this employee"""
        try:
            return self.assignments.filter_by(status='Active').all()
        except Exception as e:
            logger.error(f"Error getting active assignments for employee {self.id}: {e}")
            return []
    
    def get_workload_percentage(self):
        """Calculate current workload percentage"""
        try:
            active_assignments = self.get_active_assignments()
            return sum(assignment.allocation_percentage or 0 for assignment in active_assignments)
        except Exception as e:
            logger.error(f"Error calculating workload for employee {self.id}: {e}")
            return 0
    
    def is_available_for_project(self, project):
        """Check if employee is available for a specific project"""
        try:
            if not getattr(self, 'is_active', True):
                return False
            
            # Check overall availability
            if self.availability == 'Busy':
                return False
            
            # Check workload capacity
            current_workload = self.get_workload_percentage()
            if current_workload >= 100:
                return False
            
            # Check for date conflicts
            active_assignments = self.get_active_assignments()
            for assignment in active_assignments:
                if assignment.project and project.is_overlapping(assignment.project):
                    if (assignment.allocation_percentage or 0) >= 50:  # Major conflict
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking availability for employee {self.id}: {e}")
            return True  # Default to available if error
    
    def update_availability(self):
        """Update availability based on current workload"""
        try:
            workload = self.get_workload_percentage()
            if workload >= 100:
                self.availability = 'Busy'
            elif workload >= 50:
                self.availability = 'Partially Available'
            else:
                self.availability = 'Available'
        except Exception as e:
            logger.error(f"Error updating availability for employee {self.id}: {e}")
    
    def validate_skills_format(self, skills_data):
        if not isinstance(skills_data, list):
            return False
        for skill in skills_data:
            if not isinstance(skill, dict) or 'name' not in skill or 'proficiency' not in skill:
                return False
            if not isinstance(skill['proficiency'], int) or skill['proficiency'] < 1 or skill['proficiency'] > 10:
                return False
        return True

class Project(db.Model):
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, unique=True, index=True)
    description = db.Column(db.Text, nullable=False)
    required_skills = db.Column(db.Text, nullable=False)  # JSON string
    priority = db.Column(db.String(20), default='Medium', index=True)  # Low, Medium, High, Critical
    start_date = db.Column(db.Date, nullable=False, index=True)
    end_date = db.Column(db.Date, nullable=False, index=True)
    status = db.Column(db.String(20), default='Planning', index=True)  # Planning, Active, Completed, On Hold, Cancelled
    team_size = db.Column(db.Integer, default=3)
    assigned_employees = db.Column(db.Text, default='[]')  # JSON list of employee IDs
    budget = db.Column(db.Float, default=0.0)
    actual_cost = db.Column(db.Float, default=0.0)
    progress_percentage = db.Column(db.Integer, default=0)
    client_name = db.Column(db.String(100))
    department = db.Column(db.String(50), index=True)
    is_active = db.Column(db.Boolean, default=True, index=True)
    created_by = db.Column(db.Integer)  # Could be foreign key to User table
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    assignments = db.relationship('ProjectAssignment', backref='project', lazy='dynamic', cascade='all, delete-orphan')

    def get_required_skills(self):
        try:
            return json.loads(self.required_skills) if self.required_skills else []
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Invalid JSON in required_skills for project {self.id}")
            return []
    
    def get_assigned_employees(self):
        """Get list of assigned employee IDs from both JSON field and active assignments"""
        # Get from active assignments (more reliable)
        active_assignments = self.get_active_assignments()
        assignment_emp_ids = [a.employee_id for a in active_assignments if a.employee_id]
        
        # Also check JSON field for backwards compatibility
        try:
            json_emp_ids = json.loads(self.assigned_employees) if self.assigned_employees else []
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Invalid JSON in assigned_employees for project {self.id}")
            json_emp_ids = []
        
        # Combine and deduplicate
        all_emp_ids = list(set(assignment_emp_ids + json_emp_ids))
        
        # Update JSON field to match active assignments
        if set(json_emp_ids) != set(assignment_emp_ids):
            try:
                self.assigned_employees = json.dumps(assignment_emp_ids)
            except Exception as e:
                logger.error(f"Error updating assigned_employees JSON for project {self.id}: {e}")
        
        return assignment_emp_ids
    
    def get_active_assignments(self):
        """Get active assignments for this project"""
        try:
            return self.assignments.filter_by(status='Active').all()
        except Exception as e:
            logger.error(f"Error getting active assignments for project {self.id}: {e}")
            return []
    
    def get_team_members(self):
        """Get list of employees currently assigned to this project"""
        try:
            assignments = self.get_active_assignments()
            team_members = [assignment.employee for assignment in assignments if assignment.employee and getattr(assignment.employee, 'is_active', True)]
            return team_members
        except Exception as e:
            logger.error(f"Error getting team members for project {self.id}: {e}")
            return []
    
    def can_assign_employee(self, employee):
        """Check if an employee can be assigned to this project"""
        try:
            if not employee or not getattr(employee, 'is_active', True):
                return False, "Employee is not active"
            
            # Check if already assigned (but allow for re-assignment/updates)
            assigned_employees = self.get_assigned_employees()
            if employee.id in assigned_employees:
                return True, "Employee is already assigned to this project"  # Allow updates
            
            # Check team size limit (only for new assignments)
            current_team_size = len(self.get_team_members())
            if current_team_size >= self.team_size:
                return False, f"Project team is full ({current_team_size}/{self.team_size})"
            
            # Check employee availability
            if not employee.is_available_for_project(self):
                return False, "Employee is not available for this project"
            
            return True, "Employee can be assigned"
        except Exception as e:
            logger.error(f"Error checking if employee can be assigned to project {self.id}: {e}")
            return False, "Error checking assignment eligibility"
    
    def validate_date_range(self):
        return self.start_date <= self.end_date
    
    def is_overlapping(self, other_project):
        if not other_project:
            return False
        return not (self.end_date < other_project.start_date or self.start_date > other_project.end_date)
    
    def get_estimated_cost(self):
        """Calculate estimated cost based on assigned employees' hourly rates"""
        try:
            active_assignments = self.get_active_assignments()
            total_cost = 0.0
            
            if not active_assignments:
                return 0.0
            
            # Calculate duration in hours (assuming 8 hours per day, 5 days per week)
            duration_days = (self.end_date - self.start_date).days if self.end_date and self.start_date else 0
            duration_hours = duration_days * 8 * (5/7)  # Work days only
            
            for assignment in active_assignments:
                if assignment.employee and assignment.employee.hourly_rate:
                    allocation = (assignment.allocation_percentage or 100) / 100.0
                    employee_hours = duration_hours * allocation
                    employee_cost = employee_hours * assignment.employee.hourly_rate
                    total_cost += employee_cost
            
            return round(total_cost, 2)
        except Exception as e:
            logger.error(f"Error calculating estimated cost for project {self.id}: {e}")
            return 0.0
    
    def get_actual_cost(self):
        """Get actual cost from actual_cost field or calculate from assignments"""
        try:
            if self.actual_cost and self.actual_cost > 0:
                return self.actual_cost
            
            # Fallback to estimated cost if no actual cost recorded
            return self.get_estimated_cost()
        except Exception as e:
            logger.error(f"Error getting actual cost for project {self.id}: {e}")
            return 0.0

class ProjectAssignment(db.Model):
    __tablename__ = 'project_assignments'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False, index=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False, index=True)
    role = db.Column(db.String(100), nullable=False, default='Team Member')
    match_score = db.Column(db.Float, default=0.0)
    allocation_percentage = db.Column(db.Integer, default=100)  # Percentage of time allocated
    hourly_rate = db.Column(db.Float, default=0.0)
    estimated_hours = db.Column(db.Float, default=0.0)
    actual_hours = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='Active')  # Active, Completed, Removed
    assigned_date = db.Column(db.DateTime, default=datetime.utcnow)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints
    __table_args__ = (db.UniqueConstraint('project_id', 'employee_id', name='unique_project_employee'),)
    
    def update_from_project_dates(self):
        """Update assignment dates based on project dates"""
        if self.project:
            if not self.start_date:
                self.start_date = self.project.start_date
            if not self.end_date:
                self.end_date = self.project.end_date
    
    def get_cost(self):
        """Calculate cost for this assignment"""
        try:
            if self.actual_hours and self.actual_hours > 0:
                # Use actual hours and rate if available
                rate = self.hourly_rate or (self.employee.hourly_rate if self.employee else 0)
                return round(self.actual_hours * rate, 2)
            
            if self.estimated_hours and self.estimated_hours > 0:
                # Use estimated hours
                rate = self.hourly_rate or (self.employee.hourly_rate if self.employee else 0)
                return round(self.estimated_hours * rate, 2)
            
            # Calculate based on project duration and allocation
            if self.project and self.employee:
                duration_days = (self.project.end_date - self.project.start_date).days if self.project.end_date and self.project.start_date else 0
                duration_hours = duration_days * 8 * (5/7)  # Work days only
                allocation = (self.allocation_percentage or 100) / 100.0
                employee_hours = duration_hours * allocation
                rate = self.hourly_rate or self.employee.hourly_rate or 0
                return round(employee_hours * rate, 2)
            
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating cost for assignment {self.id}: {e}")
            return 0.0

# AI Matching Algorithm
class ProjectMatcher:
    def __init__(self):
        self.skill_weights = {
            'exact_match': 10,
            'related_match': 6,
            'experience_bonus': 5,
            'availability_bonus': 8,
            'workload_penalty': -3
        }
    
    def calculate_match_score(self, employee, project):
        """Calculate how well an employee matches a project"""
        if not employee or not project:
            return 0
            
        score = 0
        employee_skills = employee.get_skills_list()
        required_skills = project.get_required_skills()
        
        if not employee_skills or not required_skills:
            return 0
        
        # Skills matching with importance weighting
        emp_skill_names = [skill['name'].lower() for skill in employee_skills if 'name' in skill]
        total_importance_weight = 0
        matched_importance_weight = 0
        
        for req_skill in required_skills:
            if 'name' not in req_skill:
                continue
                
            req_skill_lower = req_skill['name'].lower()
            importance_multiplier = self._get_importance_multiplier(req_skill.get('importance', 'Medium'))
            total_importance_weight += importance_multiplier
            
            # Exact skill match
            exact_match = next((s for s in employee_skills if s.get('name', '').lower() == req_skill_lower), None)
            if exact_match and 'proficiency' in exact_match:
                proficiency_score = exact_match['proficiency'] / 10  # Normalize to 0-1
                skill_score = self.skill_weights['exact_match'] * proficiency_score * importance_multiplier
                score += skill_score
                matched_importance_weight += importance_multiplier
            else:
                # Related skill match
                for emp_skill in emp_skill_names:
                    if self._are_skills_related(emp_skill, req_skill_lower):
                        skill_score = self.skill_weights['related_match'] * importance_multiplier * 0.7
                        score += skill_score
                        matched_importance_weight += importance_multiplier * 0.7
                        break
        
        # Apply coverage penalty if many important skills are missing
        if total_importance_weight > 0:
            coverage_ratio = matched_importance_weight / total_importance_weight
            score *= coverage_ratio
        
        # Experience bonus with skill relevance
        relevant_experience = self._calculate_relevant_experience(employee, required_skills)
        if relevant_experience >= 5:
            score += self.skill_weights['experience_bonus']
        elif relevant_experience >= 2:
            score += self.skill_weights['experience_bonus'] * 0.6
        
        # Availability bonus/penalty
        if employee.availability == 'Available':
            score += self.skill_weights['availability_bonus']
        elif employee.availability == 'Partially Available':
            score += self.skill_weights['availability_bonus'] * 0.5
        
        # Current workload penalty with project overlap consideration
        current_projects = employee.get_current_projects()
        workload_penalty = self._calculate_workload_penalty(employee, project, current_projects)
        score += workload_penalty
        
        return max(0, score)  # Ensure non-negative score
    
    def _get_importance_multiplier(self, importance):
        """Convert importance level to numerical multiplier"""
        importance_map = {
            'critical': 3.0,
            'high': 2.0,
            'medium': 1.0,
            'low': 0.5
        }
        return importance_map.get(importance.lower(), 1.0)
    
    def _calculate_relevant_experience(self, employee, required_skills):
        """Calculate years of relevant experience based on skill overlap"""
        employee_skills = employee.get_skills_list()
        skill_names = [skill.get('name', '').lower() for skill in employee_skills]
        required_names = [skill.get('name', '').lower() for skill in required_skills]
        
        overlap_count = len(set(skill_names) & set(required_names))
        if overlap_count == 0:
            return 0
        
        # Scale experience based on skill relevance
        relevance_factor = min(1.0, overlap_count / len(required_names))
        return employee.experience_years * relevance_factor
    
    def _calculate_workload_penalty(self, employee, project, current_projects):
        """Calculate workload penalty considering project overlaps"""
        if len(current_projects) < 2:
            return 0
        
        # Check for actual date overlaps with current projects
        overlapping_projects = 0
        for proj_id in current_projects:
            existing_project = Project.query.get(proj_id)
            if existing_project and project.is_overlapping(existing_project):
                overlapping_projects += 1
        
        return self.skill_weights['workload_penalty'] * overlapping_projects
    
    def _are_skills_related(self, skill1, skill2):
        """Enhanced heuristic to determine if skills are related"""
        related_groups = [
            ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
            ['javascript', 'react', 'vue', 'angular', 'node', 'typescript', 'js'],
            ['java', 'spring', 'hibernate', 'maven', 'gradle'],
            ['sql', 'mysql', 'postgresql', 'database', 'mongodb', 'redis'],
            ['aws', 'azure', 'cloud', 'devops', 'docker', 'kubernetes', 'terraform'],
            ['ui', 'ux', 'design', 'figma', 'photoshop', 'sketch', 'adobe'],
            ['machine learning', 'ai', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn'],
            ['project management', 'scrum', 'agile', 'leadership', 'kanban'],
            ['testing', 'qa', 'selenium', 'cypress', 'junit', 'automation'],
            ['mobile', 'ios', 'android', 'react native', 'flutter', 'swift', 'kotlin']
        ]
        
        # Check for exact matches in related groups
        for group in related_groups:
            skill1_matches = [keyword for keyword in group if keyword in skill1.lower()]
            skill2_matches = [keyword for keyword in group if keyword in skill2.lower()]
            if skill1_matches and skill2_matches:
                return True
        
        # Check for partial string matches
        return skill1 in skill2 or skill2 in skill1
    
    def find_best_matches(self, project, num_matches=5, exclude_busy=True, min_score=0):
        """Find the best employee matches for a project with enhanced filtering"""
        try:
            # Use fallback query if is_active field doesn't exist
            try:
                query = Employee.query.filter_by(is_active=True)
            except:
                query = Employee.query
            
            if exclude_busy:
                query = query.filter(Employee.availability != 'Busy')
            
            # Get currently assigned employees to this project
            current_assignments = project.get_active_assignments()
            assigned_employee_ids = [a.employee_id for a in current_assignments]
            
            # Filter by department if project has one
            project_dept = getattr(project, 'department', None)
            if project_dept:
                # Prefer same department but don't exclude others
                same_dept_employees = query.filter(Employee.department == project_dept).all()
                other_dept_employees = query.filter(Employee.department != project_dept).all()
                employees = same_dept_employees + other_dept_employees
            else:
                employees = query.all()
            
            matches = []
            
            for employee in employees:
                try:
                    # Check if employee can be assigned (this includes checking if already assigned)
                    can_assign, reason = project.can_assign_employee(employee)
                    if not can_assign and employee.id not in assigned_employee_ids:
                        continue
                        
                    score = self.calculate_match_score(employee, project)
                    if score > min_score or employee.id in assigned_employee_ids:  # Include assigned employees regardless of score
                        match_data = {
                            'employee': employee,
                            'score': score,
                            'percentage': min(100, (score / 80) * 100),  # Adjusted max score
                            'match_reasons': self._get_match_reasons(employee, project, score),
                            'workload_percentage': employee.get_workload_percentage(),
                            'department_match': employee.department == getattr(project, 'department', ''),
                            'availability_status': employee.availability,
                            'estimated_cost': self._estimate_assignment_cost(employee, project),
                            'is_assigned': employee.id in assigned_employee_ids
                        }
                        matches.append(match_data)
                except Exception as e:
                    logger.error(f"Error processing employee {employee.id} for matching: {e}")
                    continue
            
            # Sort: assigned employees first, then by score descending, then by availability, then by department match
            matches.sort(key=lambda x: (not x['is_assigned'], -x['score'], x['availability_status'] != 'Available', not x['department_match']))
            
            return matches[:num_matches]
        except Exception as e:
            logger.error(f"Error finding matches for project {project.id}: {e}")
            return []
    
    def _estimate_assignment_cost(self, employee, project):
        """Estimate cost for assigning employee to project"""
        try:
            hourly_rate = getattr(employee, 'hourly_rate', 0)
            if hourly_rate <= 0:
                return 0
            
            # Estimate based on project duration and typical allocation
            project_days = (project.end_date - project.start_date).days + 1
            estimated_hours = project_days * 6  # Assume 6 hours per day average
            
            return hourly_rate * estimated_hours
        except Exception as e:
            logger.error(f"Error estimating assignment cost: {e}")
            return 0
    
    def _get_match_reasons(self, employee, project, score):
        """Generate human-readable reasons for the match"""
        reasons = []
        employee_skills = employee.get_skills_list()
        required_skills = project.get_required_skills()
        
        # Check for exact skill matches
        emp_skill_names = [skill.get('name', '').lower() for skill in employee_skills]
        for req_skill in required_skills:
            req_name = req_skill.get('name', '').lower()
            if req_name in emp_skill_names:
                skill_data = next(s for s in employee_skills if s.get('name', '').lower() == req_name)
                proficiency = skill_data.get('proficiency', 0)
                reasons.append(f"Strong in {req_skill['name']} ({proficiency}/10)")
        
        # Experience level
        if employee.experience_years >= 5:
            reasons.append(f"Senior level ({employee.experience_years} years experience)")
        elif employee.experience_years >= 2:
            reasons.append(f"Mid-level ({employee.experience_years} years experience)")
        
        # Availability
        if employee.availability == 'Available':
            reasons.append("Fully available")
        elif employee.availability == 'Partially Available':
            reasons.append("Partially available")
        
        return reasons[:3]  # Return top 3 reasons

# Gemini AI Project Matcher
class AIProjectMatcher:
    def __init__(self):
        self.gemini_model = None
        self.ai_provider = None
        self.ai_enabled = False
        self._cache = {}  # Simple in-memory cache
        self._cache_ttl = 1800  # 30 minutes
        
        # Try to initialize AI provider
        self._initialize_ai_provider()
        
        # Fallback to original matcher
        self.fallback_matcher = ProjectMatcher()
    
    def _initialize_ai_provider(self):
        """Initialize Gemini AI provider"""
        
        # Try Google Gemini first (free tier available)
        gemini_key = os.environ.get('GOOGLE_API_KEY')
        if gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                
                # Try different Gemini models in order of preference
                model_names = [
                    'gemini-1.5-flash',      # Latest fast model (free tier)
                    'gemini-1.5-pro',       # Latest pro model  
                    'gemini-pro',           # Legacy model
                    'models/gemini-1.5-flash',  # With models/ prefix
                    'models/gemini-pro'     # Legacy with prefix
                ]
                
                for model_name in model_names:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        self.ai_provider = 'gemini'
                        self.ai_enabled = True
                        logger.info(f"Google Gemini AI initialized successfully with model: {model_name}")
                        return
                    except Exception as model_error:
                        logger.debug(f"Model {model_name} failed: {model_error}")
                        continue
                
                # If no models worked
                raise Exception("No Gemini models available")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        # No AI provider available
        logger.warning("Gemini AI not available. Using algorithm-based matching.")
        self.ai_enabled = False
    
    def _get_cache_key(self, employee_id, project_id, employee_skills, project_skills):
        """Generate cache key for employee-project match"""
        data = f"{employee_id}:{project_id}:{str(employee_skills)}:{str(project_skills)}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key):
        """Get cached AI result if still valid"""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:  # Check if still valid
                return result
            else:
                del self._cache[cache_key]  # Remove expired cache
        return None
    
    def _cache_result(self, cache_key, result):
        """Cache AI result with timestamp"""
        self._cache[cache_key] = (result, time.time())
        # Simple cache cleanup - remove oldest entries if cache gets too large
        if len(self._cache) > 100:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    def analyze_project_employee_match(self, employee, project):
        """Use AI to analyze employee-project compatibility"""
        # Generate cache key
        cache_key = self._get_cache_key(
            employee.id, project.id,
            employee.get_skills_list(),
            project.get_required_skills()
        )
        
        # Check cache first
        if self.ai_enabled:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached AI result for employee {employee.id} and project {project.id}")
                return cached_result
        
        if not self.ai_enabled:
            score = self.fallback_matcher.calculate_match_score(employee, project)
            reasons = self.fallback_matcher._get_match_reasons(employee, project, score)
            return {
                'score': score,
                'reasons': reasons,
                'concerns': [],
                'ai_powered': False,
                'ai_provider': 'algorithm'
            }
        
        try:
            # Prepare employee data
            employee_skills = employee.get_skills_list()
            employee_data = {
                'name': employee.name,
                'department': employee.department,
                'skills': employee_skills,
                'experience_years': employee.experience_years,
                'availability': employee.availability,
                'current_projects': len(json.loads(employee.current_projects or '[]'))
            }
            
            # Prepare project data
            required_skills = project.get_required_skills()
            project_data = {
                'name': project.name,
                'description': project.description,
                'required_skills': required_skills,
                'priority': project.priority,
                'status': project.status,
                'budget': project.budget,
                'duration': (project.end_date - project.start_date).days if project.end_date and project.start_date else None
            }
            
            # Create prompt for AI analysis
            prompt = f"""
            Analyze the compatibility between this employee and project for team assignment.
            
            Employee Profile:
            - Name: {employee_data['name']}
            - Department: {employee_data['department']}
            - Experience: {employee_data['experience_years']} years
            - Availability: {employee_data['availability']}
            - Current Projects: {employee_data['current_projects']}
            - Skills: {json.dumps(employee_skills, indent=2)}
            
            Project Requirements:
            - Name: {project_data['name']}
            - Description: {project_data['description']}
            - Priority: {project_data['priority']}
            - Budget: ${project_data['budget']:,.2f}
            - Duration: {project_data['duration']} days
            - Required Skills: {json.dumps(required_skills, indent=2)}
            
            Please provide:
            1. A compatibility score from 0-100
            2. Top 3 reasons why this is a good match
            3. Any concerns or gaps
            
            Respond in JSON format:
            {{
                "score": <0-100>,
                "reasons": ["reason1", "reason2", "reason3"],
                "concerns": ["concern1", "concern2"] or []
            }}
            """
            
            # Get AI analysis from Gemini
            if self.ai_provider == 'gemini':
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text
            else:
                raise Exception("No AI provider available")
            
            # Parse AI response
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    ai_analysis = json.loads(json_match.group())
                else:
                    # Fallback parsing if no JSON found
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, create a structured response from text
                ai_analysis = self._parse_text_response(response_text)
            
            # Convert AI's 0-100 score to match original system's scale
            normalized_score = (ai_analysis['score'] / 100.0) * 50  # Scale to ~50 max like original
            
            result = {
                'score': normalized_score,
                'reasons': ai_analysis.get('reasons', []),
                'concerns': ai_analysis.get('concerns', []),
                'ai_powered': True,
                'ai_provider': self.ai_provider
            }
            
            # Cache the result
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Gemini AI analysis failed: {str(e)}")
            # Fallback to original algorithm
            score = self.fallback_matcher.calculate_match_score(employee, project)
            reasons = self.fallback_matcher._get_match_reasons(employee, project, score)
            return {
                'score': score,
                'reasons': reasons,
                'concerns': [],
                'ai_powered': False,
                'ai_provider': 'algorithm_fallback'
            }
    
    def _parse_text_response(self, response_text):
        """Parse unstructured AI response into structured format"""
        try:
            # Extract score using regex
            import re
            score_match = re.search(r'score[:\s]*(\d+)', response_text, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 75
            
            # Extract reasons (look for numbered lists or bullet points)
            reasons = []
            reason_patterns = [
                r'reasons?[:\s]*\n?[\s]*[1-3][\.\)]\s*([^\n]+)',
                r'good match[:\s]*\n?[\s]*[-•]\s*([^\n]+)',
                r'why[:\s]*\n?[\s]*[-•]\s*([^\n]+)'
            ]
            
            for pattern in reason_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                reasons.extend(matches[:3])
                if len(reasons) >= 3:
                    break
            
            # Extract concerns
            concerns = []
            concern_patterns = [
                r'concerns?[:\s]*\n?[\s]*[-•]\s*([^\n]+)',
                r'gaps?[:\s]*\n?[\s]*[-•]\s*([^\n]+)',
                r'issues?[:\s]*\n?[\s]*[-•]\s*([^\n]+)'
            ]
            
            for pattern in concern_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                concerns.extend(matches[:2])
                if len(concerns) >= 2:
                    break
            
            return {
                'score': min(100, max(0, score)),
                'reasons': reasons[:3] if reasons else ["Good skills match", "Relevant experience", "Available for project"],
                'concerns': concerns[:2] if concerns else []
            }
            
        except Exception as e:
            logger.error(f"Error parsing AI text response: {e}")
            return {
                'score': 75,
                'reasons': ["Skills alignment detected", "Experience level appropriate", "Available for assignment"],
                'concerns': []
            }
    
    def calculate_match_score(self, employee, project):
        """Maintain compatibility with original interface"""
        result = self.analyze_project_employee_match(employee, project)
        return result['score']
    
    def get_match_reasons(self, employee, project):
        """Get AI-powered match reasons"""
        result = self.analyze_project_employee_match(employee, project)
        return result['reasons']
    
    def find_best_matches(self, project, num_matches=8, exclude_busy=True):
        """Find best employee matches for a project using AI analysis"""
        if not project:
            return []
        
        try:
            # Get current assignments to track who's already assigned
            current_assignments = project.get_active_assignments()
            assigned_employee_ids = [assignment.employee.id for assignment in current_assignments if assignment.employee]
            
            # Base query for active employees
            query = Employee.query.filter_by(is_active=True)
            
            # Optionally exclude busy employees (but not if they're already assigned)
            if exclude_busy:
                query = query.filter(
                    (Employee.availability != 'Busy') | 
                    (Employee.id.in_(assigned_employee_ids))
                )
            
            employees = query.all()
            matches = []
            
            # Process employees in batches for better performance
            def analyze_employee(employee):
                try:
                    # Check if employee can be assigned
                    can_assign, reason = project.can_assign_employee(employee)
                    if not can_assign and employee.id not in assigned_employee_ids:
                        return None
                    
                    # Get AI analysis
                    analysis = self.analyze_project_employee_match(employee, project)
                    
                    if analysis['score'] > 0 or employee.id in assigned_employee_ids:
                        return {
                            'employee': employee,
                            'score': analysis['score'],
                            'percentage': min(100, (analysis['score'] / 50) * 100),  # Scale to 100%
                            'reasons': analysis['reasons'],
                            'concerns': analysis.get('concerns', []),
                            'ai_powered': analysis['ai_powered'],
                            'ai_provider': analysis.get('ai_provider', 'algorithm'),
                            'workload_percentage': employee.get_workload_percentage(),
                            'department_match': employee.department == getattr(project, 'department', ''),
                            'availability_status': employee.availability,
                            'estimated_cost': self._estimate_assignment_cost(employee, project),
                            'is_assigned': employee.id in assigned_employee_ids,
                            'match_reasons': analysis['reasons']  # For backward compatibility
                        }
                    return None
                except Exception as e:
                    logger.error(f"Error analyzing employee {employee.id}: {e}")
                    return None
            
            # Use ThreadPoolExecutor for concurrent AI analysis if AI is enabled
            matches = []
            if self.ai_enabled and len(employees) > 5:
                logger.info(f"Processing {len(employees)} employees with concurrent AI analysis")
                with ThreadPoolExecutor(max_workers=3) as executor:  # Limit to 3 concurrent requests
                    future_to_employee = {executor.submit(analyze_employee, emp): emp for emp in employees}
                    for future in as_completed(future_to_employee):
                        result = future.result()
                        if result is not None:
                            matches.append(result)
            else:
                # Sequential processing for smaller batches or when AI is disabled
                for employee in employees:
                    result = analyze_employee(employee)
                    if result is not None:
                        matches.append(result)
            
            # Sort: assigned employees first, then by score descending
            matches.sort(key=lambda x: (not x['is_assigned'], -x['score'], x['availability_status'] != 'Available'))
            
            return matches[:num_matches]
            
        except Exception as e:
            logger.error(f"Error finding AI matches for project {project.id}: {e}")
            # Fallback to original matcher
            return self.fallback_matcher.find_best_matches(project, num_matches, exclude_busy)
    
    def _estimate_assignment_cost(self, employee, project):
        """Estimate cost for assigning employee to project"""
        try:
            hourly_rate = getattr(employee, 'hourly_rate', 0)
            if hourly_rate <= 0:
                return 0
            
            # Estimate based on project duration and typical allocation
            project_days = (project.end_date - project.start_date).days + 1
            estimated_hours = project_days * 6  # Assume 6 hours per day average
            
            return hourly_rate * estimated_hours
        except Exception as e:
            logger.error(f"Error estimating assignment cost: {e}")
            return 0

# Initialize AI-powered matcher
matcher = AIProjectMatcher()

# Routes
@app.route('/')
def index():
    try:
        # Get separate counts to avoid join issues
        total_employees = Employee.query.count()
        total_projects = Project.query.count()
        active_projects = Project.query.filter_by(status='Active').count()
        available_employees = Employee.query.filter_by(availability='Available').count()
        
        # Get recent projects with better query
        recent_projects = Project.query.order_by(Project.created_at.desc()).limit(3).all()
        
        # Calculate overdue projects
        overdue_projects = Project.query.filter(
            and_(Project.status == 'Active', Project.end_date < datetime.now().date())
        ).count()
        
        return render_template('index.html', 
                             total_employees=total_employees,
                             total_projects=total_projects,
                             active_projects=active_projects,
                             available_employees=available_employees,
                             recent_projects=recent_projects,
                             overdue_projects=overdue_projects)
    
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return render_template('index.html', 
                             total_employees=0,
                             total_projects=0,
                             active_projects=0,
                             available_employees=0,
                             recent_projects=[],
                             overdue_projects=0)

@app.route('/employees')
def employees():
    try:
        # Add filtering and sorting capabilities
        department = request.args.get('department')
        availability = request.args.get('availability')
        sort_by = request.args.get('sort', 'name')
        
        query = Employee.query
        
        if department:
            query = query.filter(Employee.department == department)
        if availability:
            query = query.filter(Employee.availability == availability)
        
        # Apply sorting
        if sort_by == 'experience':
            query = query.order_by(Employee.experience_years.desc())
        elif sort_by == 'department':
            query = query.order_by(Employee.department, Employee.name)
        else:
            query = query.order_by(Employee.name)
        
        employees = query.all()
        
        # Get available departments and availability statuses for filtering
        departments = db.session.query(Employee.department).distinct().all()
        departments = [d[0] for d in departments]
        
        availability_statuses = ['Available', 'Partially Available', 'Busy']
        
        return render_template('employees.html', 
                             employees=employees,
                             departments=departments,
                             availability_statuses=availability_statuses,
                             current_department=department,
                             current_availability=availability,
                             current_sort=sort_by)
        
    except Exception as e:
        logger.error(f"Error loading employees: {e}")
        flash('An error occurred while loading employees.', 'error')
        return render_template('employees.html', employees=[])

def validate_employee_data(form_data):
    """Validate employee form data"""
    errors = []
    
    # Required fields
    required_fields = ['name', 'email', 'department', 'experience_years', 'availability']
    for field in required_fields:
        if not form_data.get(field, '').strip():
            errors.append(f"{field.replace('_', ' ').title()} is required")
    
    # Email validation
    email = form_data.get('email', '').strip()
    if email and not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
        errors.append("Invalid email format")
    
    # Experience years validation
    try:
        exp_years = int(form_data.get('experience_years', 0))
        if exp_years < 0 or exp_years > 50:
            errors.append("Experience years must be between 0 and 50")
    except (ValueError, TypeError):
        errors.append("Experience years must be a valid number")
    
    # Availability validation
    valid_availability = ['Available', 'Partially Available', 'Busy']
    if form_data.get('availability') not in valid_availability:
        errors.append("Invalid availability status")
    
    return errors

def parse_skills_input(skills_input):
    """Parse and validate skills input"""
    skills_list = []
    errors = []
    
    if not skills_input.strip():
        return skills_list, errors
    
    for skill_item in skills_input.split(','):
        skill_item = skill_item.strip()
        if not skill_item:
            continue
            
        if ':' in skill_item:
            try:
                name, proficiency = skill_item.split(':', 1)
                name = name.strip()
                proficiency = int(proficiency.strip())
                
                if not name:
                    errors.append(f"Empty skill name in '{skill_item}'")
                    continue
                
                if proficiency < 1 or proficiency > 10:
                    errors.append(f"Proficiency for '{name}' must be between 1 and 10")
                    continue
                
                skills_list.append({
                    'name': name,
                    'proficiency': proficiency
                })
            except ValueError:
                errors.append(f"Invalid proficiency format in '{skill_item}'")
        else:
            name = skill_item.strip()
            if name:
                skills_list.append({
                    'name': name,
                    'proficiency': 5
                })
    
    return skills_list, errors

@app.route('/employees/add', methods=['GET', 'POST'])
def add_employee():
    if request.method == 'POST':
        try:
            # Validate input data
            validation_errors = validate_employee_data(request.form)
            if validation_errors:
                for error in validation_errors:
                    flash(error, 'error')
                return render_template('add_employee.html')
            
            # Process skills input - handle both textarea and individual fields
            skills_input = request.form.get('skills', '').strip()
            
            # If textarea is empty, try to build from individual skill fields
            if not skills_input:
                skill_names = request.form.getlist('skill_names[]')
                skill_proficiencies = request.form.getlist('skill_proficiencies[]')
                skills_parts = []
                for name, prof in zip(skill_names, skill_proficiencies):
                    if name and name.strip():
                        prof_value = prof if prof else '5'
                        skills_parts.append(f"{name.strip()}:{prof_value}")
                skills_input = ', '.join(skills_parts)
            
            skills_list, skill_errors = parse_skills_input(skills_input)
            
            if skill_errors:
                for error in skill_errors:
                    flash(error, 'error')
                return render_template('add_employee.html')
            
            # Check for duplicate email (case-insensitive)
            existing_employee = Employee.query.filter(
                db.func.lower(Employee.email) == request.form['email'].strip().lower()
            ).first()
            if existing_employee:
                flash('Employee with this email already exists', 'error')
                return render_template('add_employee.html')
            
            # Process numeric fields with error handling
            try:
                experience_years = int(request.form['experience_years'])
            except (ValueError, TypeError):
                flash('Invalid experience years', 'error')
                return render_template('add_employee.html')
            
            try:
                hourly_rate = float(request.form.get('hourly_rate') or 0)
                if hourly_rate < 0:
                    flash('Hourly rate cannot be negative', 'error')
                    return render_template('add_employee.html')
            except (ValueError, TypeError):
                flash('Invalid hourly rate format', 'error')
                return render_template('add_employee.html')
            
            employee = Employee(
                name=request.form['name'].strip(),
                email=request.form['email'].strip().lower(),
                department=request.form['department'].strip(),
                skills=json.dumps(skills_list),
                experience_years=experience_years,
                availability=request.form['availability'],
                hourly_rate=hourly_rate,
                phone=request.form.get('phone', '').strip()
            )
            
            db.session.add(employee)
            db.session.commit()
            flash('Employee added successfully!', 'success')
            return redirect(url_for('employees'))
            
        except Exception as e:
            logger.error(f"Error adding employee: {e}")
            db.session.rollback()
            flash('An error occurred while adding the employee. Please try again.', 'error')
    
    return render_template('add_employee.html')

@app.route('/employees/<int:employee_id>/delete', methods=['POST'])
def delete_employee(employee_id):
    """Delete an employee and handle all related assignments"""
    try:
        employee = Employee.query.get_or_404(employee_id)
        
        # Check if employee has any active assignments
        active_assignments = ProjectAssignment.query.filter_by(
            employee_id=employee_id,
            status='Active'
        ).all()
        
        if active_assignments:
            project_names = [assignment.project.name for assignment in active_assignments if assignment.project]
            flash(f'Cannot delete {employee.name}. They are currently assigned to: {", ".join(project_names)}. Please remove them from these projects first.', 'error')
            return redirect(url_for('employees'))
        
        # Get employee name for success message
        employee_name = employee.name
        
        # Delete all assignment records (completed/inactive ones)
        ProjectAssignment.query.filter_by(employee_id=employee_id).delete()
        
        # Delete the employee
        db.session.delete(employee)
        db.session.commit()
        
        flash(f'Employee {employee_name} has been deleted successfully.', 'success')
        logger.info(f"Employee {employee_name} (ID: {employee_id}) deleted successfully")
        
    except Exception as e:
        logger.error(f"Error deleting employee {employee_id}: {e}")
        db.session.rollback()
        flash('An error occurred while deleting the employee. Please try again.', 'error')
    
    return redirect(url_for('employees'))

@app.route('/employees/<int:employee_id>/edit', methods=['GET', 'POST'])
def edit_employee(employee_id):
    """Edit an existing employee"""
    employee = Employee.query.get_or_404(employee_id)
    
    if request.method == 'POST':
        try:
            # Validate required fields
            if not request.form.get('name', '').strip():
                flash('Employee name is required.', 'error')
                return render_template('edit_employee.html', employee=employee)
            
            if not request.form.get('email', '').strip():
                flash('Employee email is required.', 'error')
                return render_template('edit_employee.html', employee=employee)
            
            # Check email uniqueness (exclude current employee)
            email = request.form['email'].strip().lower()
            existing_employee = Employee.query.filter(
                Employee.email == email,
                Employee.id != employee_id
            ).first()
            
            if existing_employee:
                flash('An employee with this email already exists.', 'error')
                return render_template('edit_employee.html', employee=employee)
            
            # Parse skills from form
            skills_list = []
            skill_names = request.form.getlist('skill_names[]')
            skill_proficiencies = request.form.getlist('skill_proficiencies[]')
            
            for name, proficiency in zip(skill_names, skill_proficiencies):
                if name and name.strip():
                    try:
                        prof_int = int(proficiency)
                        if 1 <= prof_int <= 10:
                            skills_list.append({
                                'name': name.strip(),
                                'proficiency': prof_int
                            })
                    except ValueError:
                        continue
            
            if not skills_list:
                flash('At least one skill is required.', 'error')
                return render_template('edit_employee.html', employee=employee)
            
            # Update employee data
            employee.name = request.form['name'].strip()
            employee.email = email
            employee.department = request.form['department'].strip()
            employee.skills = json.dumps(skills_list)
            employee.experience_years = int(request.form['experience_years'])
            employee.availability = request.form['availability']
            employee.hourly_rate = float(request.form.get('hourly_rate', 0))
            employee.phone = request.form.get('phone', '').strip()
            employee.updated_at = datetime.utcnow()
            
            db.session.commit()
            flash(f'Employee {employee.name} updated successfully!', 'success')
            return redirect(url_for('employees'))
            
        except Exception as e:
            logger.error(f"Error updating employee {employee_id}: {e}")
            db.session.rollback()
            flash('An error occurred while updating the employee. Please try again.', 'error')
    
    return render_template('edit_employee.html', employee=employee)

@app.route('/projects')
def projects():
    try:
        # Add filtering and sorting capabilities
        status = request.args.get('status')
        priority = request.args.get('priority')
        sort_by = request.args.get('sort', 'created_at')
        
        query = Project.query
        
        if status:
            query = query.filter(Project.status == status)
        if priority:
            query = query.filter(Project.priority == priority)
        
        # Apply sorting
        if sort_by == 'start_date':
            query = query.order_by(Project.start_date.desc())
        elif sort_by == 'priority':
            priority_order = db.case(
                (Project.priority == 'Critical', 1),
                (Project.priority == 'High', 2),
                (Project.priority == 'Medium', 3),
                (Project.priority == 'Low', 4)
            )
            query = query.order_by(priority_order)
        elif sort_by == 'name':
            query = query.order_by(Project.name)
        else:
            query = query.order_by(Project.created_at.desc())
        
        projects = query.all()
        
        # Get available statuses and priorities for filtering
        statuses = ['Planning', 'Active', 'Completed', 'On Hold']
        priorities = ['Low', 'Medium', 'High', 'Critical']
        
        return render_template('projects.html', 
                             projects=projects,
                             statuses=statuses,
                             priorities=priorities,
                             current_status=status,
                             current_priority=priority,
                             current_sort=sort_by)
        
    except Exception as e:
        logger.error(f"Error loading projects: {e}")
        flash('An error occurred while loading projects.', 'error')
        return render_template('projects.html', projects=[])

def validate_project_data(form_data):
    """Validate project form data"""
    errors = []
    
    # Required fields
    required_fields = ['name', 'description', 'priority', 'start_date', 'end_date', 'team_size']
    for field in required_fields:
        if not form_data.get(field, '').strip():
            errors.append(f"{field.replace('_', ' ').title()} is required")
    
    # Date validation
    try:
        start_date = datetime.strptime(form_data.get('start_date', ''), '%Y-%m-%d').date()
        end_date = datetime.strptime(form_data.get('end_date', ''), '%Y-%m-%d').date()
        
        if start_date > end_date:
            errors.append("Start date must be before end date")
        
        if start_date < datetime.now().date():
            errors.append("Start date cannot be in the past")
            
    except ValueError:
        errors.append("Invalid date format")
    
    # Team size validation
    try:
        team_size = int(form_data.get('team_size', 0))
        if team_size < 1 or team_size > 50:
            errors.append("Team size must be between 1 and 50")
    except (ValueError, TypeError):
        errors.append("Team size must be a valid number")
    
    # Priority validation
    valid_priorities = ['Low', 'Medium', 'High', 'Critical']
    if form_data.get('priority') not in valid_priorities:
        errors.append("Invalid priority level")
    
    return errors

def parse_required_skills(skills_input):
    """Parse and validate required skills input"""
    skills_list = []
    errors = []
    valid_importance = ['Low', 'Medium', 'High', 'Critical']
    
    if not skills_input.strip():
        return skills_list, errors
    
    for skill_item in skills_input.split(','):
        skill_item = skill_item.strip()
        if not skill_item:
            continue
            
        if ':' in skill_item:
            try:
                name, importance = skill_item.split(':', 1)
                name = name.strip()
                importance = importance.strip()
                
                if not name:
                    errors.append(f"Empty skill name in '{skill_item}'")
                    continue
                
                if importance not in valid_importance:
                    errors.append(f"Invalid importance '{importance}' for skill '{name}'. Must be one of: {', '.join(valid_importance)}")
                    continue
                
                skills_list.append({
                    'name': name,
                    'importance': importance
                })
            except ValueError:
                errors.append(f"Invalid format in '{skill_item}'")
        else:
            name = skill_item.strip()
            if name:
                skills_list.append({
                    'name': name,
                    'importance': 'Medium'
                })
    
    return skills_list, errors

@app.route('/projects/add', methods=['GET', 'POST'])
def add_project():
    if request.method == 'POST':
        try:
            # Validate input data
            validation_errors = validate_project_data(request.form)
            if validation_errors:
                for error in validation_errors:
                    flash(error, 'error')
                return render_template('add_project.html')
            
            # Process required skills - handle both textarea and individual fields
            skills_input = request.form.get('required_skills', '').strip()
            
            # If textarea is empty, try to build from individual skill fields
            if not skills_input:
                skill_names = request.form.getlist('skill_names[]')
                skill_importance = request.form.getlist('skill_importance[]')
                skills_parts = []
                for name, importance in zip(skill_names, skill_importance):
                    if name and name.strip():
                        importance_value = importance if importance else 'Medium'
                        skills_parts.append(f"{name.strip()}:{importance_value}")
                skills_input = ', '.join(skills_parts)
            
            skills_list, skill_errors = parse_required_skills(skills_input)
            
            if skill_errors:
                for error in skill_errors:
                    flash(error, 'error')
                return render_template('add_project.html')
            
            # Check for duplicate project name
            existing_project = Project.query.filter_by(name=request.form['name'].strip()).first()
            if existing_project:
                flash('Project with this name already exists', 'error')
                return render_template('add_project.html')
            
            # Process and validate fields with proper error handling
            try:
                start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d').date()
                end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d').date()
            except ValueError:
                flash('Invalid date format', 'error')
                return render_template('add_project.html')
            
            try:
                budget = float(request.form.get('budget') or 0)
                if budget < 0:
                    flash('Budget cannot be negative', 'error')
                    return render_template('add_project.html')
            except (ValueError, TypeError):
                flash('Invalid budget format', 'error')
                return render_template('add_project.html')
            
            try:
                team_size = int(request.form['team_size'])
            except (ValueError, TypeError):
                flash('Invalid team size', 'error')
                return render_template('add_project.html')
            
            project = Project(
                name=request.form['name'].strip(),
                description=request.form['description'].strip(),
                required_skills=json.dumps(skills_list),
                priority=request.form['priority'],
                start_date=start_date,
                end_date=end_date,
                team_size=team_size,
                budget=budget
            )
            
            db.session.add(project)
            db.session.commit()
            flash('Project added successfully!', 'success')
            return redirect(url_for('projects'))
            
        except Exception as e:
            logger.error(f"Error adding project: {e}")
            db.session.rollback()
            flash('An error occurred while adding the project. Please try again.', 'error')
    
    return render_template('add_project.html')

@app.route('/projects/<int:project_id>/edit', methods=['GET', 'POST'])
def edit_project(project_id):
    """Edit an existing project"""
    project = Project.query.get_or_404(project_id)
    
    if request.method == 'POST':
        try:
            # Validate required fields
            if not request.form.get('name', '').strip():
                flash('Project name is required.', 'error')
                return render_template('edit_project.html', project=project)
            
            if not request.form.get('description', '').strip():
                flash('Project description is required.', 'error')
                return render_template('edit_project.html', project=project)
            
            # Parse required skills from form
            skills_list = []
            skill_names = request.form.getlist('skill_names[]')
            skill_importance = request.form.getlist('skill_importance[]')
            
            for name, importance in zip(skill_names, skill_importance):
                if name and name.strip() and importance:
                    skills_list.append({
                        'name': name.strip(),
                        'importance': importance
                    })
            
            if not skills_list:
                flash('At least one required skill is needed.', 'error')
                return render_template('edit_project.html', project=project)
            
            # Update project data
            project.name = request.form['name'].strip()
            project.description = request.form['description'].strip()
            project.required_skills = json.dumps(skills_list)
            project.priority = request.form['priority']
            project.start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d').date()
            project.end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d').date()
            project.team_size = int(request.form['team_size'])
            project.budget = float(request.form.get('budget', 0))
            project.status = request.form.get('status', project.status)
            project.updated_at = datetime.utcnow()
            
            # Validate dates
            if project.end_date <= project.start_date:
                flash('End date must be after start date.', 'error')
                return render_template('edit_project.html', project=project)
            
            db.session.commit()
            flash(f'Project {project.name} updated successfully!', 'success')
            return redirect(url_for('projects'))
            
        except Exception as e:
            logger.error(f"Error updating project {project_id}: {e}")
            db.session.rollback()
            flash('An error occurred while updating the project. Please try again.', 'error')
    
    return render_template('edit_project.html', project=project)

@app.route('/projects/<int:project_id>/delete', methods=['POST'])
def delete_project(project_id):
    """Delete a project and handle all related assignments"""
    try:
        project = Project.query.get_or_404(project_id)
        
        # Check if project has any active assignments
        active_assignments = ProjectAssignment.query.filter_by(
            project_id=project_id,
            status='Active'
        ).all()
        
        if active_assignments:
            employee_names = [assignment.employee.name for assignment in active_assignments if assignment.employee]
            flash(f'Cannot delete {project.name}. It has active assignments with: {", ".join(employee_names)}. Please remove all employees from this project first.', 'error')
            return redirect(url_for('projects'))
        
        # Get project name for success message
        project_name = project.name
        
        # Delete all assignment records (completed/inactive ones)
        ProjectAssignment.query.filter_by(project_id=project_id).delete()
        
        # Delete the project
        db.session.delete(project)
        db.session.commit()
        
        flash(f'Project {project_name} has been deleted successfully.', 'success')
        logger.info(f"Project {project_name} (ID: {project_id}) deleted successfully")
        
    except Exception as e:
        logger.error(f"Error deleting project {project_id}: {e}")
        db.session.rollback()
        flash('An error occurred while deleting the project. Please try again.', 'error')
    
    return redirect(url_for('projects'))

@app.route('/projects/<int:project_id>/status', methods=['POST'])
def update_project_status(project_id):
    """Quick update project status"""
    try:
        project = Project.query.get_or_404(project_id)
        new_status = request.form.get('status')
        
        if new_status not in ['Planning', 'Active', 'On Hold', 'Completed', 'Cancelled']:
            flash('Invalid status provided.', 'error')
            return redirect(url_for('projects'))
        
        old_status = project.status
        project.status = new_status
        project.updated_at = datetime.utcnow()
        
        # If project is being completed, update all active assignments
        if new_status == 'Completed' and old_status != 'Completed':
            active_assignments = ProjectAssignment.query.filter_by(
                project_id=project_id,
                status='Active'
            ).all()
            
            for assignment in active_assignments:
                assignment.status = 'Completed'
                assignment.end_date = datetime.utcnow().date()
        
        db.session.commit()
        flash(f'Project {project.name} status updated to {new_status}.', 'success')
        logger.info(f"Project {project.name} (ID: {project_id}) status updated from {old_status} to {new_status}")
        
    except Exception as e:
        logger.error(f"Error updating project status {project_id}: {e}")
        db.session.rollback()
        flash('An error occurred while updating the project status.', 'error')
    
    return redirect(url_for('projects'))

@app.route('/projects/<int:project_id>/match')
def match_employees(project_id):
    try:
        project = Project.query.get_or_404(project_id)
        
        # Get number of matches from query parameter, default to 6 for faster loading
        num_matches = min(15, max(1, int(request.args.get('num_matches', 6))))
        
        # Check if we should exclude busy employees
        exclude_busy = request.args.get('exclude_busy', 'true').lower() == 'true'
        
        # Get current assignments
        current_assignments = project.get_active_assignments()
        assigned_employees = [assignment.employee for assignment in current_assignments if assignment.employee]
        assigned_employee_ids = [emp.id for emp in assigned_employees]
        
        # Get matches
        matches = matcher.find_best_matches(project, num_matches=num_matches, exclude_busy=exclude_busy)
        
        # Mark already assigned employees in the matches
        for match in matches:
            match['is_assigned'] = match['employee'].id in assigned_employee_ids
        
        return render_template('project_matches.html', 
                             project=project, 
                             matches=matches,
                             assigned_employees=assigned_employees,
                             assigned_employee_ids=assigned_employee_ids,
                             num_matches=num_matches,
                             exclude_busy=exclude_busy)
        
    except Exception as e:
        logger.error(f"Error matching employees for project {project_id}: {e}")
        flash('An error occurred while finding employee matches.', 'error')
        return redirect(url_for('projects'))

@app.route('/projects/<int:project_id>/assign', methods=['POST'])
def assign_employees(project_id):
    try:
        project = Project.query.get_or_404(project_id)
        selected_employee_ids = request.form.getlist('employee_ids')
        
        # Get currently assigned employees
        current_assignments = project.get_active_assignments()
        currently_assigned_ids = [a.employee_id for a in current_assignments]
        
        # If no employees selected, this might be an "unassign all" action
        if not selected_employee_ids:
            if currently_assigned_ids:
                # Remove all current assignments
                for assignment in current_assignments:
                    assignment.status = 'Removed'
                    if assignment.employee:
                        # Update employee's current projects
                        current_projects = assignment.employee.get_current_projects()
                        if project_id in current_projects:
                            current_projects.remove(project_id)
                            assignment.employee.current_projects = json.dumps(current_projects)
                        assignment.employee.update_availability()
                
                # Update project
                project.assigned_employees = json.dumps([])
                if project.status == 'Active':
                    project.status = 'Planning'
                
                db.session.commit()
                flash('All employees removed from project', 'success')
            else:
                flash('Please select at least one employee', 'error')
            return redirect(url_for('match_employees', project_id=project_id))
        
        # Validate employee IDs
        valid_employee_ids = []
        for emp_id in selected_employee_ids:
            try:
                emp_id = int(emp_id)
                employee = Employee.query.get(emp_id)
                if employee:
                    valid_employee_ids.append(emp_id)
                else:
                    flash(f'Employee with ID {emp_id} not found', 'error')
            except ValueError:
                flash(f'Invalid employee ID: {emp_id}', 'error')
        
        if not valid_employee_ids:
            flash('No valid employees selected', 'error')
            return redirect(url_for('match_employees', project_id=project_id))
        
        # Get existing assigned employees and merge with new selections
        all_assigned_ids = list(set(currently_assigned_ids + valid_employee_ids))
        
        # Check if assigning too many employees
        if len(all_assigned_ids) > project.team_size:
            flash(f'Cannot assign {len(all_assigned_ids)} employees to a project with team size {project.team_size}. Currently assigned: {len(currently_assigned_ids)}, New selections: {len(valid_employee_ids)}', 'error')
            return redirect(url_for('match_employees', project_id=project_id))
        
        # Process new assignments
        new_assignments = 0
        for emp_id in valid_employee_ids:
            if emp_id not in currently_assigned_ids:
                employee = Employee.query.get(emp_id)
                
                # Check if employee can be assigned
                can_assign, reason = project.can_assign_employee(employee)
                if not can_assign:
                    flash(f'Cannot assign {employee.name}: {reason}', 'error')
                    continue
                
                # Check if there's an existing assignment (any status) for this employee and project
                existing_assignment = ProjectAssignment.query.filter_by(
                    project_id=project_id,
                    employee_id=emp_id
                ).first()
                
                if existing_assignment:
                    # Reactivate existing assignment instead of creating new one
                    if existing_assignment.status != 'Active':
                        existing_assignment.status = 'Active'
                        existing_assignment.assigned_date = datetime.utcnow()
                        existing_assignment.match_score = matcher.calculate_match_score(employee, project)
                        existing_assignment.allocation_percentage = 100
                        existing_assignment.hourly_rate = getattr(employee, 'hourly_rate', 0.0)
                        existing_assignment.update_from_project_dates()
                        new_assignments += 1
                        logger.info(f"Reactivated existing assignment for employee {emp_id} to project {project_id}")
                else:
                    # Create new assignment record
                    assignment = ProjectAssignment(
                        project_id=project_id,
                        employee_id=emp_id,
                        role='Team Member',
                        match_score=matcher.calculate_match_score(employee, project),
                        allocation_percentage=100,
                        hourly_rate=getattr(employee, 'hourly_rate', 0.0)
                    )
                    assignment.update_from_project_dates()
                    db.session.add(assignment)
                    new_assignments += 1
                    logger.info(f"Created new assignment for employee {emp_id} to project {project_id}")
                
                # Update employee's current projects
                current_projects = employee.get_current_projects()
                if project_id not in current_projects:
                    current_projects.append(project_id)
                    employee.current_projects = json.dumps(current_projects)
                
                # Update employee availability
                employee.update_availability()
        
        # Update project with all assigned employees (existing + new)
        project.assigned_employees = json.dumps(all_assigned_ids)
        if project.status == 'Planning' and all_assigned_ids:
            project.status = 'Active'
        
        db.session.commit()
        
        if new_assignments > 0:
            flash(f'Successfully assigned {new_assignments} new employees to {project.name}! Total team members: {len(all_assigned_ids)}', 'success')
        else:
            flash(f'No new employees were assigned. Total team members: {len(all_assigned_ids)}', 'info')
        
        return redirect(url_for('match_employees', project_id=project_id))
        
    except Exception as e:
        logger.error(f"Error assigning employees to project {project_id}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Handle the specific "transaction has been rolled back" error
        try:
            db.session.rollback()
        except Exception as rollback_error:
            logger.error(f"Error during rollback: {rollback_error}")
            # Remove the session if rollback fails
            db.session.remove()
        
        # Provide more user-friendly error messages
        if "UNIQUE constraint failed" in str(e):
            flash('This employee is already assigned to this project. Please refresh the page and try again.', 'error')
        elif "transaction has been rolled back" in str(e):
            flash('Database transaction error. Please try again.', 'error')
        else:
            flash(f'An error occurred while assigning employees: {str(e)}', 'error')
        
        return redirect(url_for('match_employees', project_id=project_id))

@app.route('/schedule')
def schedule():
    try:
        active_projects = Project.query.filter(Project.status.in_(['Active', 'Planning'])).order_by(Project.start_date).all()
        schedule_data = []
        
        for project in active_projects:
            assigned_emp_ids = project.get_assigned_employees()
            assigned_employees = [Employee.query.get(emp_id) for emp_id in assigned_emp_ids if Employee.query.get(emp_id)]
            
            # Calculate project progress
            total_days = (project.end_date - project.start_date).days
            if project.status == 'Active':
                days_elapsed = (datetime.now().date() - project.start_date).days
                progress_percentage = max(0, min(100, (days_elapsed / total_days) * 100))
            else:
                progress_percentage = 0
            
            schedule_data.append({
                'project': project,
                'employees': assigned_employees,
                'duration_days': total_days,
                'progress_percentage': progress_percentage,
                'days_remaining': max(0, (project.end_date - datetime.now().date()).days),
                'is_overdue': datetime.now().date() > project.end_date and project.status == 'Active'
            })
        
        return render_template('schedule.html', schedule_data=schedule_data)
        
    except Exception as e:
        logger.error(f"Error loading schedule: {e}")
        flash('An error occurred while loading the schedule.', 'error')
        return redirect(url_for('index'))

@app.route('/api/employee/<int:employee_id>')
def api_employee_details(employee_id):
    try:
        employee = Employee.query.get_or_404(employee_id)
        current_projects = employee.get_current_projects()
        
        # Get current project details
        current_project_details = []
        for proj_id in current_projects:
            project = Project.query.get(proj_id)
            if project:
                current_project_details.append({
                    'id': project.id,
                    'name': project.name,
                    'status': project.status,
                    'end_date': project.end_date.isoformat()
                })
        
        return jsonify({
            'id': employee.id,
            'name': employee.name,
            'email': employee.email,
            'department': employee.department,
            'skills': employee.get_skills_list(),
            'experience_years': employee.experience_years,
            'availability': employee.availability,
            'current_projects': current_project_details,
            'workload': len(current_projects)
        })
        
    except Exception as e:
        logger.error(f"Error fetching employee details for ID {employee_id}: {e}")
        return jsonify({'error': 'Failed to fetch employee details'}), 500

def init_sample_data():
    """Initialize the database with sample employees and projects"""
    # Sample employees
    sample_employees = [
        {
            'name': 'Alice Johnson',
            'email': 'alice.johnson@company.com',
            'department': 'Engineering',
            'skills': [
                {'name': 'Python', 'proficiency': 9},
                {'name': 'Django', 'proficiency': 8},
                {'name': 'SQL', 'proficiency': 7},
                {'name': 'Machine Learning', 'proficiency': 6}
            ],
            'experience_years': 5,
            'availability': 'Available'
        },
        {
            'name': 'Bob Smith',
            'email': 'bob.smith@company.com',
            'department': 'Engineering',
            'skills': [
                {'name': 'JavaScript', 'proficiency': 9},
                {'name': 'React', 'proficiency': 8},
                {'name': 'Node.js', 'proficiency': 7},
                {'name': 'UI/UX Design', 'proficiency': 6}
            ],
            'experience_years': 4,
            'availability': 'Available'
        },
        {
            'name': 'Carol Davis',
            'email': 'carol.davis@company.com',
            'department': 'Design',
            'skills': [
                {'name': 'UI/UX Design', 'proficiency': 9},
                {'name': 'Figma', 'proficiency': 8},
                {'name': 'Photoshop', 'proficiency': 7},
                {'name': 'User Research', 'proficiency': 8}
            ],
            'experience_years': 6,
            'availability': 'Partially Available'
        },
        {
            'name': 'David Wilson',
            'email': 'david.wilson@company.com',
            'department': 'DevOps',
            'skills': [
                {'name': 'AWS', 'proficiency': 8},
                {'name': 'Docker', 'proficiency': 9},
                {'name': 'Kubernetes', 'proficiency': 7},
                {'name': 'Python', 'proficiency': 6}
            ],
            'experience_years': 7,
            'availability': 'Available'
        },
        {
            'name': 'Emma Brown',
            'email': 'emma.brown@company.com',
            'department': 'Data Science',
            'skills': [
                {'name': 'Python', 'proficiency': 9},
                {'name': 'Machine Learning', 'proficiency': 8},
                {'name': 'TensorFlow', 'proficiency': 7},
                {'name': 'SQL', 'proficiency': 8}
            ],
            'experience_years': 3,
            'availability': 'Available'
        },
        {
            'name': 'Frank Miller',
            'email': 'frank.miller@company.com',
            'department': 'Management',
            'skills': [
                {'name': 'Project Management', 'proficiency': 9},
                {'name': 'Scrum', 'proficiency': 8},
                {'name': 'Leadership', 'proficiency': 8},
                {'name': 'Strategic Planning', 'proficiency': 7}
            ],
            'experience_years': 10,
            'availability': 'Busy'
        },
        {
            'name': 'Grace Lee',
            'email': 'grace.lee@company.com',
            'department': 'Engineering',
            'skills': [
                {'name': 'Java', 'proficiency': 8},
                {'name': 'Spring', 'proficiency': 7},
                {'name': 'SQL', 'proficiency': 8},
                {'name': 'API Design', 'proficiency': 7}
            ],
            'experience_years': 4,
            'availability': 'Available'
        },
        {
            'name': 'Henry Chen',
            'email': 'henry.chen@company.com',
            'department': 'QA',
            'skills': [
                {'name': 'Test Automation', 'proficiency': 8},
                {'name': 'Selenium', 'proficiency': 7},
                {'name': 'Python', 'proficiency': 6},
                {'name': 'Quality Assurance', 'proficiency': 9}
            ],
            'experience_years': 5,
            'availability': 'Available'
        }
    ]
    
    # Add employees if they don't exist
    for emp_data in sample_employees:
        if not Employee.query.filter_by(email=emp_data['email']).first():
            employee = Employee(
                name=emp_data['name'],
                email=emp_data['email'],
                department=emp_data['department'],
                skills=json.dumps(emp_data['skills']),
                experience_years=emp_data['experience_years'],
                availability=emp_data['availability']
            )
            db.session.add(employee)
    
    # Sample projects
    sample_projects = [
        {
            'name': 'E-commerce Mobile App',
            'description': 'Develop a cross-platform mobile application for online shopping with AI recommendations.',
            'required_skills': [
                {'name': 'React', 'importance': 'High'},
                {'name': 'JavaScript', 'importance': 'High'},
                {'name': 'UI/UX Design', 'importance': 'Medium'},
                {'name': 'Machine Learning', 'importance': 'Medium'}
            ],
            'priority': 'High',
            'start_date': datetime.now().date(),
            'end_date': (datetime.now() + timedelta(days=90)).date(),
            'team_size': 4
        },
        {
            'name': 'Data Analytics Dashboard',
            'description': 'Create a comprehensive analytics dashboard for business intelligence and reporting.',
            'required_skills': [
                {'name': 'Python', 'importance': 'High'},
                {'name': 'SQL', 'importance': 'High'},
                {'name': 'Machine Learning', 'importance': 'Medium'},
                {'name': 'UI/UX Design', 'importance': 'Low'}
            ],
            'priority': 'Medium',
            'start_date': (datetime.now() + timedelta(days=30)).date(),
            'end_date': (datetime.now() + timedelta(days=120)).date(),
            'team_size': 3
        }
    ]
    
    # Add projects if they don't exist
    for proj_data in sample_projects:
        if not Project.query.filter_by(name=proj_data['name']).first():
            project = Project(
                name=proj_data['name'],
                description=proj_data['description'],
                required_skills=json.dumps(proj_data['required_skills']),
                priority=proj_data['priority'],
                start_date=proj_data['start_date'],
                end_date=proj_data['end_date'],
                team_size=proj_data['team_size']
            )
            db.session.add(project)
    
    try:
        db.session.commit()
        logger.info("Sample data initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing sample data: {e}")
        db.session.rollback()
        raise

# Additional route handlers for enhanced functionality
@app.route('/employees/<int:employee_id>')
def employee_detail(employee_id):
    try:
        employee = Employee.query.get_or_404(employee_id)
        
        # Get employee's project history
        assignments = employee.assignments.order_by(ProjectAssignment.assigned_date.desc()).all()
        
        # Get current projects
        current_assignments = [a for a in assignments if a.status == 'Active']
        
        # Calculate statistics
        total_projects = len(assignments)
        completed_projects = len([a for a in assignments if a.status == 'Completed'])
        total_hours = sum(a.actual_hours for a in assignments if a.actual_hours)
        total_earnings = sum(a.get_cost() for a in assignments)
        
        return render_template('employee_detail.html', 
                             employee=employee,
                             assignments=assignments,
                             current_assignments=current_assignments,
                             total_projects=total_projects,
                             completed_projects=completed_projects,
                             total_hours=total_hours,
                             total_earnings=total_earnings)
        
    except Exception as e:
        logger.error(f"Error loading employee details for ID {employee_id}: {e}")
        flash('Error loading employee details.', 'error')
        return redirect(url_for('employees'))

@app.route('/projects/<int:project_id>')
def project_detail(project_id):
    try:
        project = Project.query.get_or_404(project_id)
        
        # Get project assignments
        assignments = project.get_active_assignments()
        
        # Get project statistics
        total_estimated_hours = sum(a.estimated_hours for a in assignments if a.estimated_hours)
        total_actual_hours = sum(a.actual_hours for a in assignments if a.actual_hours)
        estimated_cost = project.get_estimated_cost()
        actual_cost = project.get_actual_cost()
        
        # Calculate team utilization
        team_utilization = []
        for assignment in assignments:
            if assignment.employee:
                team_utilization.append({
                    'employee': assignment.employee,
                    'allocation': assignment.allocation_percentage,
                    'workload': assignment.employee.get_workload_percentage(),
                    'role': assignment.role
                })
        
        return render_template('project_detail.html',
                             project=project,
                             assignments=assignments,
                             team_utilization=team_utilization,
                             total_estimated_hours=total_estimated_hours,
                             total_actual_hours=total_actual_hours,
                             estimated_cost=estimated_cost,
                             actual_cost=actual_cost)
        
    except Exception as e:
        logger.error(f"Error loading project details for ID {project_id}: {e}")
        flash('Error loading project details.', 'error')
        return redirect(url_for('projects'))

@app.route('/projects/<int:project_id>/remove_employee/<int:employee_id>', methods=['POST'])
def remove_employee_from_project(project_id, employee_id):
    try:
        project = Project.query.get_or_404(project_id)
        employee = Employee.query.get_or_404(employee_id)
        
        # Find the assignment
        assignment = ProjectAssignment.query.filter_by(
            project_id=project_id, 
            employee_id=employee_id, 
            status='Active'
        ).first()
        
        if assignment:
            # Update assignment status
            assignment.status = 'Removed'
            
            # Update employee's current projects
            current_projects = employee.get_current_projects()
            if project_id in current_projects:
                current_projects.remove(project_id)
                employee.current_projects = json.dumps(current_projects)
            
            # Update employee availability
            employee.update_availability()
            
            # Update project's assigned employees list
            assigned_employees = project.get_assigned_employees()
            if employee_id in assigned_employees:
                assigned_employees.remove(employee_id)
                project.assigned_employees = json.dumps(assigned_employees)
            
            db.session.commit()
            flash(f'{employee.name} removed from {project.name}', 'success')
        else:
            flash(f'{employee.name} is not assigned to {project.name}', 'error')
        
    except Exception as e:
        logger.error(f"Error removing employee from project: {e}")
        db.session.rollback()
        flash('Error removing employee from project', 'error')
    
    return redirect(url_for('match_employees', project_id=project_id))

# Additional API endpoints
@app.route('/api/skills/suggestions')
def api_skill_suggestions():
    try:
        # Get all unique skills from employees
        employees = Employee.query.all()
        all_skills = set()
        
        for emp in employees:
            skills = emp.get_skills_list()
            for skill in skills:
                if 'name' in skill:
                    all_skills.add(skill['name'])
        
        # Get all unique skills from projects
        projects = Project.query.all()
        for proj in projects:
            skills = proj.get_required_skills()
            for skill in skills:
                if 'name' in skill:
                    all_skills.add(skill['name'])
        
        return jsonify(sorted(list(all_skills)))
        
    except Exception as e:
        logger.error(f"Error getting skill suggestions: {e}")
        return jsonify([])

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('403.html'), 403

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            init_sample_data()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true', 
            port=int(os.environ.get('PORT', 5000)),
            host=os.environ.get('HOST', '127.0.0.1'))