"""Image attachment generator (PNG, JPG)"""

import io
import random
from typing import Dict, Any
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .metadata import ATTACHMENT_METADATA

class ImageGenerator:
    """Generates image attachments (PNG, JPG) with various diagram types"""
    
    def __init__(self, config, file_type: str = "png"):
        self.config = config
        self.file_type = file_type

        try:
            self.font_large = ImageFont.truetype("arial.ttf", 24)
            self.font_medium = ImageFont.truetype("arial.ttf", 16)
            self.font_small = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):

            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def generate(self, category: str) -> Dict[str, Any]:
        """Generate image based on category"""

        generators = {
            "api": self._generate_api_flow_diagram,
            "rest": self._generate_api_flow_diagram,
            "technical": self._generate_architecture_diagram,
            "architecture": self._generate_architecture_diagram,
            "project": self._generate_project_chart,
            "planning": self._generate_gantt_chart,
            "releases": self._generate_burndown_chart
        }
        
        generator_func = generators.get(category, self._generate_architecture_diagram)

        category_key = category if category in ["api", "technical", "project"] else "technical"
        filenames = ATTACHMENT_METADATA[self.file_type].get(category_key, [f"Diagram.{self.file_type}"])
        filename = f"{self.config.prefix}_{random.choice(filenames)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.file_type}"

        img_buffer = generator_func()
        
        return {
            "filename": filename,
            "content": img_buffer.getvalue(),
            "content_type": f"image/{self.file_type}",
            "size": img_buffer.tell()
        }
    
    def _generate_api_flow_diagram(self) -> io.BytesIO:
        """Generate API flow diagram"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        ax.text(5, 9.5, 'API Request Flow', fontsize=20, ha='center', weight='bold')

        components = [
            {'name': 'Client App', 'pos': (1, 7), 'color': '#3B82F6'},
            {'name': 'API Gateway', 'pos': (5, 7), 'color': '#10B981'},
            {'name': 'Auth Service', 'pos': (8, 8), 'color': '#F59E0B'},
            {'name': 'Load Balancer', 'pos': (5, 5), 'color': '#8B5CF6'},
            {'name': 'Service A', 'pos': (2, 3), 'color': '#EF4444'},
            {'name': 'Service B', 'pos': (5, 3), 'color': '#EF4444'},
            {'name': 'Service C', 'pos': (8, 3), 'color': '#EF4444'},
            {'name': 'Database', 'pos': (5, 1), 'color': '#6B7280'},
        ]

        for comp in components:
            box = FancyBboxPatch(
                (comp['pos'][0] - 0.8, comp['pos'][1] - 0.3),
                1.6, 0.6,
                boxstyle="round,pad=0.1",
                facecolor=comp['color'],
                edgecolor='black',
                alpha=0.8
            )
            ax.add_patch(box)
            ax.text(comp['pos'][0], comp['pos'][1], comp['name'], 
                   ha='center', va='center', fontsize=10, color='white', weight='bold')

        connections = [
            ((1, 7), (5, 7), '1. Request'),
            ((5, 7), (8, 8), '2. Auth'),
            ((8, 8), (5, 7), '3. Token'),
            ((5, 7), (5, 5), '4. Route'),
            ((5, 5), (2, 3), '5a. Service'),
            ((5, 5), (5, 3), '5b. Service'),
            ((5, 5), (8, 3), '5c. Service'),
            ((2, 3), (5, 1), '6. Query'),
            ((5, 3), (5, 1), '6. Query'),
            ((8, 3), (5, 1), '6. Query'),
        ]
        
        for start, end, label in connections:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))

            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y + 0.1, label, fontsize=8, ha='center')

        ax.text(0.5, 0.2, 'Flow: Client → Gateway → Auth → Services → Database', 
               fontsize=10, style='italic')

        buffer = io.BytesIO()
        plt.savefig(buffer, format=self.file_type, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        buffer.seek(0)
        return buffer
    
    def _generate_architecture_diagram(self) -> io.BytesIO:
        """Generate system architecture diagram"""
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 9)
        ax.axis('off')

        ax.text(6, 8.5, 'Microservices Architecture', fontsize=22, ha='center', weight='bold')

        layers = [
            {'name': 'Client Layer', 'y': 7, 'color': '#E0E7FF'},
            {'name': 'API Gateway', 'y': 5.5, 'color': '#C7D2FE'},
            {'name': 'Service Layer', 'y': 4, 'color': '#A5B4FC'},
            {'name': 'Data Layer', 'y': 2, 'color': '#818CF8'},
            {'name': 'Infrastructure', 'y': 0.5, 'color': '#6366F1'},
        ]

        for layer in layers:
            rect = patches.Rectangle((0.5, layer['y'] - 0.4), 11, 0.8,
                                   facecolor=layer['color'], edgecolor='black', alpha=0.5)
            ax.add_patch(rect)
            ax.text(0.2, layer['y'], layer['name'], fontsize=10, rotation=90, 
                   va='center', weight='bold')

        clients = ['Web App', 'Mobile App', 'API Client']
        for i, client in enumerate(clients):
            x = 2 + i * 3
            self._draw_component(ax, x, 7, client, '#1F2937')

        self._draw_component(ax, 6, 5.5, 'API Gateway\n(Kong)', '#059669', width=2)

        services = [
            ('Auth Service', 2, '#DC2626'),
            ('User Service', 4, '#DC2626'),
            ('Product Service', 6, '#DC2626'),
            ('Order Service', 8, '#DC2626'),
            ('Notification', 10, '#DC2626'),
        ]
        for name, x, color in services:
            self._draw_component(ax, x, 4, name, color)
        
        # Data stores
        datastores = [
            ('PostgreSQL\n(Primary)', 3, '#059669'),
            ('Redis\n(Cache)', 6, '#7C3AED'),
            ('MongoDB\n(Logs)', 9, '#EA580C'),
        ]
        for name, x, color in datastores:
            self._draw_component(ax, x, 2, name, color)
        
        # Infrastructure
        infra = ['Docker', 'Kubernetes', 'Monitoring']
        for i, comp in enumerate(infra):
            x = 2 + i * 3.5
            self._draw_component(ax, x, 0.5, comp, '#1E40AF')
        
        # Add some connections
        # Gateway to services
        for x in [2, 4, 6, 8, 10]:
            ax.plot([6, x], [5.2, 4.3], 'k--', alpha=0.5)
        
        # Services to data
        ax.plot([3, 3], [3.7, 2.3], 'k--', alpha=0.5)
        ax.plot([6, 6], [3.7, 2.3], 'k--', alpha=0.5)
        ax.plot([9, 9], [3.7, 2.3], 'k--', alpha=0.5)

        buffer = io.BytesIO()
        plt.savefig(buffer, format=self.file_type, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        buffer.seek(0)
        return buffer
    
    def _generate_project_chart(self) -> io.BytesIO:
        """Generate project status chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Sprint velocity chart
        sprints = list(range(1, 11))
        planned = [30, 32, 35, 35, 38, 40, 40, 42, 42, 45]
        actual = [28, 30, 33, 36, 35, 42, 38, 44, 41, 43]
        
        ax1.plot(sprints, planned, 'b-o', label='Planned', linewidth=2, markersize=8)
        ax1.plot(sprints, actual, 'g-s', label='Actual', linewidth=2, markersize=8)
        ax1.fill_between(sprints, actual, alpha=0.3, color='green')
        
        ax1.set_xlabel('Sprint Number', fontsize=12)
        ax1.set_ylabel('Story Points', fontsize=12)
        ax1.set_title('Sprint Velocity Trend', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.5, 10.5)
        ax1.set_ylim(20, 50)
        
        # Project status pie chart
        labels = ['Completed', 'In Progress', 'To Do', 'Blocked']
        sizes = [45, 30, 20, 5]
        colors = ['#10B981', '#3B82F6', '#9CA3AF', '#EF4444']
        explode = (0.1, 0, 0, 0.1)
        
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax2.set_title('Project Task Status', fontsize=14, weight='bold')
        
        # Overall title
        fig.suptitle('Project Dashboard - Sprint 10', fontsize=16, weight='bold')

        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format=self.file_type, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        buffer.seek(0)
        return buffer
    
    def _generate_gantt_chart(self) -> io.BytesIO:
        """Generate Gantt chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Project tasks
        tasks = [
            ('Requirements Analysis', 0, 10),
            ('System Design', 8, 12),
            ('Database Design', 10, 8),
            ('Backend Development', 15, 25),
            ('Frontend Development', 18, 22),
            ('API Integration', 30, 10),
            ('Testing Phase', 35, 15),
            ('UAT', 45, 10),
            ('Deployment Prep', 50, 5),
            ('Go Live', 55, 2),
        ]
        
        # Colors for different task types
        task_colors = {
            'Requirements Analysis': '#3B82F6',
            'System Design': '#8B5CF6',
            'Database Design': '#8B5CF6',
            'Backend Development': '#10B981',
            'Frontend Development': '#10B981',
            'API Integration': '#F59E0B',
            'Testing Phase': '#EF4444',
            'UAT': '#EF4444',
            'Deployment Prep': '#6B7280',
            'Go Live': '#1F2937',
        }
        
        # Plot tasks
        for i, (task, start, duration) in enumerate(tasks):
            ax.barh(i, duration, left=start, height=0.5, 
                   color=task_colors[task], alpha=0.8, edgecolor='black')
            
            # Add task name
            ax.text(start - 1, i, task, ha='right', va='center', fontsize=10)
            
            # Add duration text
            if duration > 5:
                ax.text(start + duration/2, i, f'{duration}d', 
                       ha='center', va='center', fontsize=9, color='white', weight='bold')
        
        # Add milestones
        milestones = [
            ('Design Complete', 20, '#DC2626'),
            ('Dev Complete', 40, '#059669'),
            ('Testing Complete', 50, '#7C3AED'),
        ]
        
        for milestone, day, color in milestones:
            ax.plot([day, day], [-0.5, len(tasks) - 0.5], color=color, 
                   linestyle='--', linewidth=2, alpha=0.7)
            ax.text(day, len(tasks), milestone, rotation=45, 
                   ha='left', va='bottom', fontsize=9, color=color)
        
        # Formatting
        ax.set_xlabel('Project Days', fontsize=12)
        ax.set_title('Project Timeline - Gantt Chart', fontsize=16, weight='bold')
        ax.set_ylim(-0.5, len(tasks) - 0.5)
        ax.set_xlim(-15, 65)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_yticks([])
        
        # Add current day marker
        current_day = 32
        ax.axvline(x=current_day, color='red', linestyle='-', linewidth=2, alpha=0.5)
        ax.text(current_day, -1, 'Today', ha='center', fontsize=10, color='red')

        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format=self.file_type, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        buffer.seek(0)
        return buffer
    
    def _generate_burndown_chart(self) -> io.BytesIO:
        """Generate sprint burndown chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sprint data (10 days)
        days = list(range(11))
        ideal_burndown = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
        actual_burndown = [100, 95, 85, 78, 65, 58, 45, 35, 25, 15, 8]
        
        # Plot lines
        ax.plot(days, ideal_burndown, 'b--', label='Ideal Burndown', linewidth=2)
        ax.plot(days, actual_burndown, 'r-o', label='Actual Burndown', linewidth=2, markersize=8)
        
        # Fill area
        ax.fill_between(days, actual_burndown, alpha=0.3, color='red')
        
        # Add annotations for key points
        ax.annotate('Sprint Start', xy=(0, 100), xytext=(0.5, 105),
                   arrowprops=dict(arrowstyle='->', color='black'))
        ax.annotate('Behind Schedule', xy=(5, 58), xytext=(6, 70),
                   arrowprops=dict(arrowstyle='->', color='red'))
        ax.annotate('8 points remaining', xy=(10, 8), xytext=(8, 20),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        # Formatting
        ax.set_xlabel('Sprint Day', fontsize=12)
        ax.set_ylabel('Story Points Remaining', fontsize=12)
        ax.set_title('Sprint 10 Burndown Chart', fontsize=16, weight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(0, 110)
        
        # Add statistics box
        stats_text = 'Sprint Statistics:\n' \
                    'Total Points: 100\n' \
                    'Completed: 92\n' \
                    'Remaining: 8\n' \
                    'Velocity: 92%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format=self.file_type, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        buffer.seek(0)
        return buffer
    
    def _draw_component(self, ax, x, y, text, color, width=1.5, height=0.5):
        """Helper to draw a component box"""
        box = FancyBboxPatch(
            (x - width/2, y - height/2),
            width, height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            alpha=0.8
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
               fontsize=9, color='white', weight='bold')