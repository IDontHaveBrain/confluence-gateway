"""Multilingual documentation content generator"""

import random
from typing import Dict, Any, List
from datetime import datetime

class MultilangDocsGenerator:
    """Generates multilingual documentation content"""
    
    def __init__(self, config, language: str = "en"):
        self.config = config
        self.language = language
        self.content_bank = {
            "en": self._english_content(),
            "ko": self._korean_content(),
            "mixed": self._mixed_content()
        }
    
    def generate(self, category: str, space_type: str) -> Dict[str, Any]:
        """Generate multilingual documentation content"""
        content_templates = self.content_bank.get(self.language, self.content_bank["en"])
        template = random.choice(content_templates)

        topic = random.choice(["Cloud Migration", "Security Policy", "Data Processing", "System Architecture", "API Integration"])
        
        title = template["title_pattern"].format(
            topic=topic,
            version=random.choice(["v1.0", "v2.0", "2024"]),
            type=template["doc_type"]
        )
        
        content = self._generate_content(template, topic)
        labels = ["multilingual", self.language, category, space_type.lower()]
        
        return {
            "title": title,
            "content": content,
            "labels": labels,
            "metadata": {
                "category": category,
                "language": self.language,
                "type": "multilingual",
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _generate_content(self, template: Dict[str, Any], topic: str) -> str:
        """Generate HTML content based on language template"""
        html_parts = []

        if self.language == "mixed":
            html_parts.append("""
                <ac:structured-macro ac:name="info">
                    <ac:rich-text-body>
                        <p>This document contains both English and Korean content / 이 문서는 영어와 한국어 내용을 모두 포함합니다</p>
                    </ac:rich-text-body>
                </ac:structured-macro>
            """)

        for section in template["sections"]:
            html_parts.append(self._generate_section(section, topic))

        if self.config.search_optimization["semantic_pairs"]:
            html_parts.append(self._add_multilingual_semantic_content())
        
        return "\n".join(html_parts)
    
    def _generate_section(self, section: Dict[str, Any], topic: str) -> str:
        """Generate a section based on language"""
        html = f"<h2>{section['heading']}</h2>"
        
        if section["type"] == "parallel":

            html += """
                <table>
                    <thead>
                        <tr>
                            <th>English</th>
                            <th>한국어</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for content_pair in section["content"]:
                html += f"""
                        <tr>
                            <td>{content_pair['en']}</td>
                            <td>{content_pair['ko']}</td>
                        </tr>
                """
            html += "</tbody></table>"
            
        elif section["type"] == "terminology":
            html += self._generate_terminology_table(section["terms"])
            
        elif section["type"] == "content":
            for paragraph in section["paragraphs"]:
                html += f"<p>{paragraph}</p>"
                
        elif section["type"] == "code_example":
            html += f"""
                <ac:structured-macro ac:name="code">
                    <ac:parameter ac:name="language">{section.get('language', 'python')}</ac:parameter>
                    <ac:plain-text-body><![CDATA[{section['code']}]]></ac:plain-text-body>
                </ac:structured-macro>
            """
            if "explanation" in section:
                html += f"<p>{section['explanation']}</p>"
        
        return html
    
    def _generate_terminology_table(self, terms: List[Dict[str, str]]) -> str:
        """Generate terminology comparison table"""
        html = """
            <table>
                <thead>
                    <tr>
                        <th>English Term</th>
                        <th>Korean Term (한국어 용어)</th>
                        <th>Description (설명)</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for term in terms:
            html += f"""
                    <tr>
                        <td>{term['en']}</td>
                        <td>{term['ko']}</td>
                        <td>{term['description']}</td>
                    </tr>
            """
        
        html += "</tbody></table>"
        return html
    
    def _add_multilingual_semantic_content(self) -> str:
        """Add content optimized for multilingual semantic search"""
        if self.language == "en":
            return """
                <h3>Keywords for Search</h3>
                <p>This document covers implementation details, configuration settings, deployment procedures, 
                troubleshooting guides, and best practices. It includes technical specifications, 
                system requirements, and operational guidelines.</p>
            """
        elif self.language == "ko":
            return """
                <h3>검색 키워드</h3>
                <p>이 문서는 구현 세부사항, 구성 설정, 배포 절차, 문제 해결 가이드, 모범 사례를 다룹니다. 
                기술 사양, 시스템 요구사항, 운영 지침이 포함되어 있습니다.</p>
            """
        else:  # mixed
            return """
                <h3>Search Keywords / 검색 키워드</h3>
                <p>Implementation (구현), Configuration (설정), Deployment (배포), 
                Troubleshooting (문제 해결), Best Practices (모범 사례), 
                Technical Specifications (기술 사양), System Requirements (시스템 요구사항)</p>
            """
    
    def _english_content(self) -> List[Dict[str, Any]]:
        """English content templates"""
        return [
            {
                "title_pattern": "{topic} Implementation Guide - {version}",
                "doc_type": "Implementation Guide",
                "sections": [
                    {
                        "heading": "Overview",
                        "type": "content",
                        "paragraphs": [
                            "This guide provides comprehensive instructions for implementing {topic} in enterprise environments. "
                            "It covers architecture decisions, security considerations, and deployment strategies.",
                            "The document is intended for system architects, DevOps engineers, and technical leads "
                            "responsible for planning and executing large-scale deployments."
                        ]
                    },
                    {
                        "heading": "Technical Requirements",
                        "type": "content",
                        "paragraphs": [
                            "Before starting the implementation, ensure your environment meets the following requirements:",
                            "• Minimum 16GB RAM for production deployments",
                            "• High-availability database cluster with automatic failover",
                            "• Load balancer with SSL termination support",
                            "• Container orchestration platform (Kubernetes recommended)"
                        ]
                    },
                    {
                        "heading": "Configuration Example",
                        "type": "code_example",
                        "language": "yaml",
                        "code": """apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgresql://localhost:5432/myapp"
  cache_enabled: "true"
  log_level: "info"
  max_connections: "100" """,
                        "explanation": "This configuration map defines essential application settings for production deployment."
                    }
                ]
            }
        ]
    
    def _korean_content(self) -> List[Dict[str, Any]]:
        """Korean content templates"""
        return [
            {
                "title_pattern": "{topic} 구현 가이드 - {version}",
                "doc_type": "구현 가이드",
                "sections": [
                    {
                        "heading": "개요",
                        "type": "content",
                        "paragraphs": [
                            "이 가이드는 엔터프라이즈 환경에서 {topic}을(를) 구현하기 위한 포괄적인 지침을 제공합니다. "
                            "아키텍처 결정사항, 보안 고려사항 및 배포 전략을 다룹니다.",
                            "이 문서는 대규모 배포를 계획하고 실행하는 시스템 아키텍트, DevOps 엔지니어 및 "
                            "기술 리더를 대상으로 합니다."
                        ]
                    },
                    {
                        "heading": "기술 요구사항",
                        "type": "content",
                        "paragraphs": [
                            "구현을 시작하기 전에 환경이 다음 요구사항을 충족하는지 확인하십시오:",
                            "• 프로덕션 배포를 위한 최소 16GB RAM",
                            "• 자동 장애 조치 기능이 있는 고가용성 데이터베이스 클러스터",
                            "• SSL 종료 지원이 있는 로드 밸런서",
                            "• 컨테이너 오케스트레이션 플랫폼 (Kubernetes 권장)"
                        ]
                    },
                    {
                        "heading": "구성 예제",
                        "type": "code_example",
                        "language": "yaml",
                        "code": """apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgresql://localhost:5432/myapp"
  cache_enabled: "true"
  log_level: "info"
  max_connections: "100" """,
                        "explanation": "이 구성 맵은 프로덕션 배포를 위한 필수 애플리케이션 설정을 정의합니다."
                    }
                ]
            }
        ]
    
    def _mixed_content(self) -> List[Dict[str, Any]]:
        """Mixed language content templates"""
        return [
            {
                "title_pattern": "{topic} Technical Reference / 기술 참조 - {version}",
                "doc_type": "Technical Reference / 기술 참조",
                "sections": [
                    {
                        "heading": "Introduction / 소개",
                        "type": "parallel",
                        "content": [
                            {
                                "en": "This technical reference provides detailed information about system components and their interactions.",
                                "ko": "이 기술 참조는 시스템 구성 요소와 상호 작용에 대한 자세한 정보를 제공합니다."
                            },
                            {
                                "en": "It serves as a comprehensive resource for developers and system administrators.",
                                "ko": "개발자와 시스템 관리자를 위한 포괄적인 리소스 역할을 합니다."
                            }
                        ]
                    },
                    {
                        "heading": "Technical Terminology / 기술 용어",
                        "type": "terminology",
                        "terms": [
                            {
                                "en": "Load Balancer",
                                "ko": "로드 밸런서",
                                "description": "Distributes network traffic across multiple servers / 여러 서버에 네트워크 트래픽을 분산"
                            },
                            {
                                "en": "Microservices",
                                "ko": "마이크로서비스",
                                "description": "Architectural style that structures an application as a collection of services / 애플리케이션을 서비스 모음으로 구성하는 아키텍처 스타일"
                            },
                            {
                                "en": "Container",
                                "ko": "컨테이너",
                                "description": "Lightweight, standalone executable package / 가볍고 독립적인 실행 가능한 패키지"
                            },
                            {
                                "en": "API Gateway",
                                "ko": "API 게이트웨이",
                                "description": "Entry point for all client requests / 모든 클라이언트 요청의 진입점"
                            },
                            {
                                "en": "Cache",
                                "ko": "캐시",
                                "description": "Temporary storage for frequently accessed data / 자주 액세스하는 데이터의 임시 저장소"
                            }
                        ]
                    },
                    {
                        "heading": "Best Practices / 모범 사례",
                        "type": "parallel",
                        "content": [
                            {
                                "en": "Always implement proper error handling and logging mechanisms",
                                "ko": "항상 적절한 오류 처리 및 로깅 메커니즘을 구현하십시오"
                            },
                            {
                                "en": "Use environment variables for configuration management",
                                "ko": "구성 관리를 위해 환경 변수를 사용하십시오"
                            },
                            {
                                "en": "Implement health checks for all services",
                                "ko": "모든 서비스에 대해 상태 확인을 구현하십시오"
                            },
                            {
                                "en": "Follow the principle of least privilege for security",
                                "ko": "보안을 위해 최소 권한 원칙을 따르십시오"
                            }
                        ]
                    },
                    {
                        "heading": "Code Example / 코드 예제",
                        "type": "code_example",
                        "language": "python",
                        "code": """# Health check endpoint / 상태 확인 엔드포인트
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }""",
                        "explanation": "Simple health check implementation / 간단한 상태 확인 구현"
                    }
                ]
            }
        ]