use super::*;
use crate::error::PipelineError;
use crate::schema::{JobControl, JobStatus};

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

// ── MockStep and MockTask for happy-path and heartbeat tests ────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MockStep {
    value: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum MockTask {
    Step1(MockStep),
    Step2(MockStep),
}

#[async_trait]
impl Task for MockTask {
    type Context = ();

    fn current_step_name(&self) -> &'static str {
        match self {
            MockTask::Step1(_) => "Step1",
            MockTask::Step2(_) => "Step2",
        }
    }

    fn validate_transition(_current: &str, _next: &str) -> Result<(), PipelineError> {
        Ok(())
    }

    async fn execute_current(
        self,
        _db: &PgPool,
        _context: &(),
        _cancel: &CancellationToken,
        _progress: &ProgressReporter,
    ) -> Result<StepResult<Self>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            MockTask::Step1(s) => Ok(StepResult::Next(MockTask::Step2(MockStep {
                value: s.value,
            }))),
            MockTask::Step2(_) => Ok(StepResult::Done),
        }
    }

    async fn on_cancel_current(
        self,
        _db: &PgPool,
        _context: &(),
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    async fn on_delete_current(
        self,
        _db: &PgPool,
        _context: &(),
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

// ── SlowMockTask for timeout test ───────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SlowMockTask {
    value: String,
}

#[async_trait]
impl Task for SlowMockTask {
    type Context = ();

    fn current_step_name(&self) -> &'static str {
        "SlowMockTask"
    }

    fn validate_transition(_current: &str, _next: &str) -> Result<(), PipelineError> {
        Ok(())
    }

    async fn execute_current(
        self,
        _db: &PgPool,
        _context: &(),
        _cancel: &CancellationToken,
        _progress: &ProgressReporter,
    ) -> Result<StepResult<Self>, Box<dyn std::error::Error + Send + Sync>> {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        Ok(StepResult::Done)
    }

    async fn on_cancel_current(
        self,
        _db: &PgPool,
        _context: &(),
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    async fn on_delete_current(
        self,
        _db: &PgPool,
        _context: &(),
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Build a JobRow with sensible defaults for testing.
fn test_job_row(step_data: serde_json::Value) -> JobRow {
    JobRow {
        id: Uuid::new_v4(),
        job_type: "test".to_string(),
        job_key: "executor-test".to_string(),
        pipeline_version: 1,
        status: JobStatus::Running,
        control: JobControl::None,
        current_step: "MockStep".to_string(),
        step_data,
        result: serde_json::Value::Object(serde_json::Map::new()),
        tried: 0,
        max_retries: 0,
        retry_delay_secs: 0,
        priority: 0,
        wakeup_at: Utc::now(),
        step_started_at: None,
        step_completed_at: None,
        timeout_at: None,
        worker_id: None,
        last_heartbeat_at: None,
        progress: None,
        error: None,
        created_by: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        completed_at: None,
    }
}

fn test_config() -> WorkerConfig {
    WorkerConfig {
        max_concurrent: 4,
        poll_interval_secs: 2,
        drain_timeout_secs: 300,
        heartbeat_interval_secs: 3600,
        zombie_threshold_secs: 120,
        recovery_interval_secs: 60,
        recovery_wakeup_secs: 5,
        cancel_poll_interval_secs: 3600,
        notify_channel: "pipeline_jobs_changed".to_string(),
        worker_id: Uuid::new_v4().to_string(),
    }
}

fn lazy_pool() -> PgPool {
    PgPool::connect_lazy("postgres://invalid:5432/none").expect("connect_lazy should not fail")
}

// ── Tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_execute_step_happy_path() {
    let db = lazy_pool();
    let config = test_config();
    let step_data =
        serde_json::to_value(MockTask::Step1(MockStep {
            value: "test".into(),
        }))
        .unwrap();
    let job = test_job_row(step_data);

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        execute_step::<MockTask>(&job, &db, &(), &config),
    )
    .await
    .expect("execute_step should complete within 2s");

    match result {
        ExecutionResult::Success(StepResult::Next(next_task), _) => {
            assert!(
                matches!(next_task, MockTask::Step2(_)),
                "expected Step2 variant"
            );
        }
        other => panic!("expected ExecutionResult::Success(Next), got {other:?}"),
    }
}

#[tokio::test]
async fn test_execute_step_timeout() {
    let db = lazy_pool();
    let config = test_config();
    let step_data = serde_json::to_value(SlowMockTask {
        value: "slow".into(),
    })
    .unwrap();
    let mut job = test_job_row(step_data);
    job.timeout_at = Some(Utc::now() + chrono::Duration::milliseconds(50));

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        execute_step::<SlowMockTask>(&job, &db, &(), &config),
    )
    .await
    .expect("execute_step should complete within 2s");

    assert!(
        matches!(result, ExecutionResult::Timeout),
        "expected Timeout, got {result:?}"
    );
}

#[tokio::test]
async fn test_execute_step_deserialization_failure() {
    let db = lazy_pool();
    let config = test_config();
    let job = test_job_row(serde_json::json!({"garbage": true}));

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        execute_step::<MockTask>(&job, &db, &(), &config),
    )
    .await
    .expect("execute_step should complete within 2s");

    match result {
        ExecutionResult::Failed(msg) => {
            assert!(
                msg.contains("deserialize"),
                "error message should mention deserialize, got: {msg}"
            );
        }
        other => panic!("expected ExecutionResult::Failed, got {other:?}"),
    }
}

#[tokio::test]
async fn test_heartbeat_aborted_after_completion() {
    let db = lazy_pool();
    let config = test_config();
    let step_data =
        serde_json::to_value(MockTask::Step2(MockStep {
            value: "done".into(),
        }))
        .unwrap();
    let job = test_job_row(step_data);

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        execute_step::<MockTask>(&job, &db, &(), &config),
    )
    .await
    .expect("execute_step should complete within 2s — heartbeat was not leaked");

    assert!(
        matches!(result, ExecutionResult::Done(_)),
        "expected Done, got {result:?}"
    );
}
