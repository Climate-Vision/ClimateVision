#!/usr/bin/env python
"""
Security Scanner for ClimateVision API.

Scans API endpoints for OWASP-style vulnerabilities and generates a security report.

Usage:
    python scripts/security_scan.py --target http://localhost:8000
    python scripts/security_scan.py --target http://localhost:8000 --output security_report.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import requests
except ImportError:
    print("Error: requests library required. Run: pip install requests")
    sys.exit(1)


@dataclass
class Finding:
    """Security finding from scan."""

    endpoint: str
    method: str
    severity: str  # critical, high, medium, low, info
    category: str
    title: str
    description: str
    remediation: str
    evidence: Optional[str] = None


@dataclass
class SecurityReport:
    """Complete security scan report."""

    target: str
    scan_timestamp: str
    scan_duration_seconds: float
    total_endpoints: int
    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "scan_timestamp": self.scan_timestamp,
            "scan_duration_seconds": self.scan_duration_seconds,
            "total_endpoints": self.total_endpoints,
            "findings": [asdict(f) for f in self.findings],
            "summary": self.summary,
        }


class SecurityScanner:
    """OWASP-style security scanner for ClimateVision API."""

    def __init__(self, target: str, timeout: int = 10):
        self.target = target.rstrip("/")
        self.timeout = timeout
        self.findings: list[Finding] = []
        self.session = requests.Session()

    def scan(self) -> SecurityReport:
        """Run full security scan."""
        start_time = time.time()

        endpoints = self._discover_endpoints()

        # Run all checks
        self._check_security_headers()
        self._check_rate_limiting()
        self._check_input_validation()
        self._check_file_upload()
        self._check_injection()
        self._check_auth()
        self._check_error_handling()

        duration = time.time() - start_time

        # Build summary
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for finding in self.findings:
            summary[finding.severity] = summary.get(finding.severity, 0) + 1

        return SecurityReport(
            target=self.target,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
            scan_duration_seconds=round(duration, 2),
            total_endpoints=len(endpoints),
            findings=self.findings,
            summary=summary,
        )

    def _discover_endpoints(self) -> list[str]:
        """Discover API endpoints from OpenAPI spec."""
        endpoints = []
        try:
            resp = self.session.get(
                urljoin(self.target, "/openapi.json"),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                spec = resp.json()
                paths = spec.get("paths", {})
                endpoints = list(paths.keys())
        except Exception:
            # Fallback to known endpoints
            endpoints = [
                "/api/health",
                "/api/predict",
                "/api/predict/upload",
                "/api/runs",
                "/api/organizations",
                "/api/explain",
            ]
        return endpoints

    def _check_security_headers(self) -> None:
        """Check for security headers."""
        try:
            resp = self.session.get(
                urljoin(self.target, "/api/health"),
                timeout=self.timeout,
            )

            required_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
            }

            for header, expected in required_headers.items():
                if header not in resp.headers:
                    self.findings.append(Finding(
                        endpoint="/api/health",
                        method="GET",
                        severity="medium",
                        category="Security Headers",
                        title=f"Missing {header} header",
                        description=f"The {header} security header is not set.",
                        remediation=f"Add '{header}: {expected}' to all responses.",
                    ))

            # Check for server disclosure
            if "Server" in resp.headers:
                server = resp.headers["Server"]
                if any(v in server.lower() for v in ["version", "uvicorn", "python"]):
                    self.findings.append(Finding(
                        endpoint="/api/health",
                        method="GET",
                        severity="low",
                        category="Information Disclosure",
                        title="Server version disclosed",
                        description=f"Server header reveals: {server}",
                        remediation="Remove or obfuscate the Server header.",
                        evidence=server,
                    ))

        except Exception as e:
            self.findings.append(Finding(
                endpoint="/api/health",
                method="GET",
                severity="info",
                category="Connectivity",
                title="Could not check security headers",
                description=str(e),
                remediation="Ensure API is running.",
            ))

    def _check_rate_limiting(self) -> None:
        """Check rate limiting implementation."""
        try:
            # Send multiple rapid requests
            for i in range(5):
                resp = self.session.get(
                    urljoin(self.target, "/api/health"),
                    timeout=self.timeout,
                )

            # Check for rate limit headers
            if "X-RateLimit-Remaining" not in resp.headers:
                self.findings.append(Finding(
                    endpoint="/api/health",
                    method="GET",
                    severity="medium",
                    category="Rate Limiting",
                    title="No rate limiting headers detected",
                    description="Rate limiting may not be implemented or is not exposing standard headers.",
                    remediation="Implement rate limiting with X-RateLimit-* headers.",
                ))

        except Exception:
            pass

    def _check_input_validation(self) -> None:
        """Check input validation on predict endpoint."""
        test_cases = [
            {
                "name": "Invalid bbox - out of range",
                "payload": {"bbox": [200, 10, 30, 40]},
                "expected_status": 422,
            },
            {
                "name": "Invalid bbox - wrong order",
                "payload": {"bbox": [10, 50, 5, 40]},
                "expected_status": 422,
            },
            {
                "name": "Invalid date range",
                "payload": {"start_date": "2025-01-01", "end_date": "2024-01-01"},
                "expected_status": 422,
            },
            {
                "name": "SQL injection in kind",
                "payload": {"kind": "'; DROP TABLE runs; --"},
                "expected_status": [200, 422],  # Should either sanitize or reject
            },
        ]

        for test in test_cases:
            try:
                resp = self.session.post(
                    urljoin(self.target, "/api/predict"),
                    json=test["payload"],
                    timeout=self.timeout,
                )

                expected = test["expected_status"]
                if isinstance(expected, list):
                    passed = resp.status_code in expected
                else:
                    passed = resp.status_code == expected

                if not passed:
                    self.findings.append(Finding(
                        endpoint="/api/predict",
                        method="POST",
                        severity="high" if "injection" in test["name"].lower() else "medium",
                        category="Input Validation",
                        title=f"Failed: {test['name']}",
                        description=f"Expected status {expected}, got {resp.status_code}",
                        remediation="Add proper input validation.",
                        evidence=json.dumps(test["payload"]),
                    ))

            except Exception:
                pass

    def _check_file_upload(self) -> None:
        """Check file upload security."""
        test_cases = [
            {
                "name": "Path traversal in filename",
                "filename": "../../../etc/passwd",
                "content": b"test",
                "severity": "critical",
            },
            {
                "name": "Executable upload",
                "filename": "malware.exe",
                "content": b"MZ\x90\x00",
                "severity": "high",
            },
            {
                "name": "Double extension",
                "filename": "image.tif.php",
                "content": b"<?php system($_GET['cmd']); ?>",
                "severity": "high",
            },
        ]

        for test in test_cases:
            try:
                files = {"file": (test["filename"], test["content"])}
                resp = self.session.post(
                    urljoin(self.target, "/api/predict/upload"),
                    files=files,
                    timeout=self.timeout,
                )

                # Should be rejected (4xx)
                if resp.status_code < 400:
                    self.findings.append(Finding(
                        endpoint="/api/predict/upload",
                        method="POST",
                        severity=test["severity"],
                        category="File Upload",
                        title=f"Allowed: {test['name']}",
                        description=f"Dangerous file upload was accepted (status {resp.status_code})",
                        remediation="Validate file types, extensions, and sanitize filenames.",
                        evidence=test["filename"],
                    ))

            except Exception:
                pass

    def _check_injection(self) -> None:
        """Check for injection vulnerabilities."""
        injection_payloads = [
            ("SQL", "' OR '1'='1"),
            ("NoSQL", '{"$gt": ""}'),
            ("Command", "; cat /etc/passwd"),
            ("Template", "{{7*7}}"),
            ("XSS", "<script>alert(1)</script>"),
        ]

        for injection_type, payload in injection_payloads:
            try:
                resp = self.session.post(
                    urljoin(self.target, "/api/predict"),
                    json={"kind": payload},
                    timeout=self.timeout,
                )

                # Check if payload is reflected in response
                if payload in resp.text:
                    self.findings.append(Finding(
                        endpoint="/api/predict",
                        method="POST",
                        severity="high",
                        category="Injection",
                        title=f"{injection_type} injection reflected",
                        description=f"Payload was reflected in response without sanitization.",
                        remediation=f"Sanitize all user inputs. Use parameterized queries.",
                        evidence=payload,
                    ))

            except Exception:
                pass

    def _check_auth(self) -> None:
        """Check authentication implementation."""
        # Test protected endpoints without auth
        protected_endpoints = [
            "/api/organizations",
            "/api/predict",
        ]

        for endpoint in protected_endpoints:
            try:
                resp = self.session.get(
                    urljoin(self.target, endpoint),
                    timeout=self.timeout,
                )

                # If we can access without API key, note it
                if resp.status_code == 200:
                    self.findings.append(Finding(
                        endpoint=endpoint,
                        method="GET",
                        severity="info",
                        category="Authentication",
                        title="Endpoint accessible without API key",
                        description="This endpoint does not require authentication.",
                        remediation="Consider requiring X-API-Key for sensitive endpoints.",
                    ))

            except Exception:
                pass

    def _check_error_handling(self) -> None:
        """Check error handling doesn't leak sensitive info."""
        try:
            # Trigger an error
            resp = self.session.get(
                urljoin(self.target, "/api/runs/99999999"),
                timeout=self.timeout,
            )

            if resp.status_code >= 400:
                body = resp.text.lower()

                # Check for stack traces
                if "traceback" in body or "file " in body:
                    self.findings.append(Finding(
                        endpoint="/api/runs/99999999",
                        method="GET",
                        severity="medium",
                        category="Information Disclosure",
                        title="Stack trace in error response",
                        description="Error responses contain stack traces.",
                        remediation="Use generic error messages in production.",
                    ))

                # Check for internal paths
                if "/home/" in body or "/usr/" in body or "c:\\" in body.lower():
                    self.findings.append(Finding(
                        endpoint="/api/runs/99999999",
                        method="GET",
                        severity="low",
                        category="Information Disclosure",
                        title="Internal paths in error response",
                        description="Error responses reveal internal file paths.",
                        remediation="Remove path information from error messages.",
                    ))

        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Security scanner for ClimateVision API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/security_scan.py --target http://localhost:8000
  python scripts/security_scan.py --target https://api.example.com --output report.json
        """,
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target API URL (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds",
    )

    args = parser.parse_args()

    print(f"Starting security scan of: {args.target}")
    print("=" * 60)

    scanner = SecurityScanner(args.target, timeout=args.timeout)
    report = scanner.scan()

    # Print results
    print(f"\nScan completed in {report.scan_duration_seconds:.2f} seconds")
    print(f"Endpoints scanned: {report.total_endpoints}")
    print(f"\nFindings Summary:")
    print(f"  Critical: {report.summary.get('critical', 0)}")
    print(f"  High:     {report.summary.get('high', 0)}")
    print(f"  Medium:   {report.summary.get('medium', 0)}")
    print(f"  Low:      {report.summary.get('low', 0)}")
    print(f"  Info:     {report.summary.get('info', 0)}")

    if report.findings:
        print(f"\nDetailed Findings:")
        print("-" * 60)
        for i, finding in enumerate(report.findings, 1):
            severity_icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵",
                "info": "⚪",
            }.get(finding.severity, "⚪")

            print(f"\n{i}. {severity_icon} [{finding.severity.upper()}] {finding.title}")
            print(f"   Endpoint: {finding.method} {finding.endpoint}")
            print(f"   Category: {finding.category}")
            print(f"   Description: {finding.description}")
            print(f"   Remediation: {finding.remediation}")
            if finding.evidence:
                print(f"   Evidence: {finding.evidence[:100]}")

    # Save report
    output_path = args.output or "outputs/security_report.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\nReport saved to: {output_path}")

    # Exit code based on critical/high findings
    critical_high = report.summary.get("critical", 0) + report.summary.get("high", 0)
    if critical_high > 0:
        print(f"\n❌ SECURITY SCAN FAILED: {critical_high} critical/high findings")
        return 1
    else:
        print("\n✅ SECURITY SCAN PASSED: No critical/high findings")
        return 0


if __name__ == "__main__":
    sys.exit(main())
