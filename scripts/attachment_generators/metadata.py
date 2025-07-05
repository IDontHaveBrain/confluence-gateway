"""Attachment metadata templates for dummy data generation"""

# Attachment metadata templates
ATTACHMENT_METADATA = {
    "pdf": {
        "api": ["API_Specification_v2.0.pdf", "API_Integration_Guide.pdf", "REST_API_Reference.pdf"],
        "technical": ["System_Architecture_Diagram.pdf", "Database_Schema.pdf", "Network_Topology.pdf"],
        "project": ["Project_Charter.pdf", "Requirements_Document.pdf", "Test_Plan.pdf"],
    },
    "docx": {
        "api": ["API_Tutorial.docx", "Authentication_Guide.docx", "Error_Codes_Reference.docx"],
        "technical": ["Installation_Manual.docx", "Configuration_Guide.docx", "Troubleshooting_Guide.docx"],
        "project": ["Meeting_Minutes.docx", "Project_Proposal.docx", "Status_Report.docx"],
    },
    "xlsx": {
        "api": ["API_Endpoints_Matrix.xlsx", "Rate_Limits_Table.xlsx", "Response_Codes.xlsx"],
        "technical": ["System_Requirements.xlsx", "Performance_Metrics.xlsx", "Capacity_Planning.xlsx"],
        "project": ["Project_Timeline.xlsx", "Resource_Allocation.xlsx", "Risk_Register.xlsx"],
    },
    "png": {
        "api": ["API_Flow_Diagram.png", "Authentication_Flow.png", "Data_Model.png"],
        "technical": ["Architecture_Overview.png", "Deployment_Diagram.png", "Component_Interaction.png"],
        "project": ["Gantt_Chart.png", "Burndown_Chart.png", "Team_Structure.png"],
    },
    "jpg": {
        "api": ["API_Dashboard_Screenshot.jpg", "Response_Example.jpg", "Error_Message.jpg"],
        "technical": ["Server_Rack_Photo.jpg", "Network_Diagram.jpg", "Console_Output.jpg"],
        "project": ["Whiteboard_Photo.jpg", "Team_Meeting.jpg", "Sprint_Board.jpg"],
    }
}