"""
Municipal Operations Platform - Integration Verification Test Client
Usage:
  1. Start the backend: python app.py
  2. In another terminal: python test_client.py
"""
import httpx
import sys

BASE_URL = "http://localhost:8000"

def run_tests():
    print("Testing connection to Municipal Backend API at", BASE_URL)
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # 1. Fetch statutory municipal departments
        res = client.get("/api/v1/admin/departments")
        if res.status_code != 200:
            print("FAILED to connect to backend. Is 'python app.py' running?")
            sys.exit(1)
        
        depts = res.json()["departments"]
        print(f"\n[OK] Loaded {len(depts)} statutory municipal departments:")
        for d in depts:
            print(f"  - [{d['code']}] {d['department_name']} -> {d['alert_channel']}")
        
        # 2. Check Admin Inbox
        inbox = client.get("/api/v1/admin/unassigned-tickets").json()
        print(f"\n[OK] Admin Inbox accessible. Total pending complaints: {inbox['total_pending_alerts']}")
        for t in inbox['tickets']:
            print(f"  - [{t['ticket_id']}] {t['detected_problem']} -> Pre-selected: {t['pre_selected_department_name']} (Risk: {t['risk_score']})")

        # 3. Check Dashboard Stats
        stats = client.get("/api/v1/admin/dashboard-stats").json()
        print(f"\n[OK] Dashboard Stats KPI:")
        print(f"  - Total Complaints: {stats['total_complaints']}")
        print(f"  - Pending Alerts: {stats['pending_alerts']}")
        print(f"  - Critical Hazards: {stats['critical_priority_hazards']}")

        # 4. Check Public Feed
        feed = client.get("/api/v1/reports").json()
        print(f"\n[OK] Public Citizen Feed: {feed['total']} reports accessible.")

        print("\nAll endpoints operational and 100% ready for frontend integration!")

if __name__ == "__main__":
    run_tests()
