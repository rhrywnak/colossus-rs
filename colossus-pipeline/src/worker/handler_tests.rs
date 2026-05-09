use super::*;
use crate::cancel::CancellationToken;
use crate::progress::ProgressReporter;
use crate::schema::{JobControl, JobStatus};

use async_trait::async_trait;
use uuid::Uuid;

// ── MockTask ────────────────────────────────────────────────────────

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

async fn test_db() -> PgPool {
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set for this test");
    PgPool::connect(&url).await.unwrap()
}

fn test_config() -> WorkerConfig {
    WorkerConfig {
        max_concurrent: 4,
        poll_interval_secs: 2,
        drain_timeout_secs: 300,
        heartbeat_interval_secs: 30,
        zombie_threshold_secs: 120,
        recovery_interval_secs: 60,
        recovery_wakeup_secs: 5,
        cancel_poll_interval_secs: 5,
        notify_channel: "pipeline_jobs_changed".to_string(),
        worker_id: Uuid::new_v4().to_string(),
    }
}

async fn insert_running_job(
    db: &PgPool,
    job_id: Uuid,
    tried: i32,
    max_retries: i32,
) -> JobRow {
    sqlx::query(
        "INSERT INTO pipeline_jobs (id, job_type, job_key, pipeline_version, \
         status, control, current_step, step_data, result, tried, max_retries, \
         retry_delay_secs, priority, wakeup_at, created_at, updated_at) \
         VALUES ($1, 'test', $2, 1, $3, $4, 'Step1', $5, '{}', $6, $7, \
         60, 0, NOW(), NOW(), NOW())",
    )
    .bind(job_id)
    .bind(format!("handler-test-{job_id}"))
    .bind(JobStatus::Running.as_str())
    .bind(JobControl::None.as_str())
    .bind(serde_json::to_value(MockTask::Step1(MockStep {
        value: "test".into(),
    }))
    .unwrap())
    .bind(tried)
    .bind(max_retries)
    .execute(db)
    .await
    .unwrap();

    sqlx::query_as::<_, JobRow>("SELECT * FROM pipeline_jobs WHERE id = $1")
        .bind(job_id)
        .fetch_one(db)
        .await
        .unwrap()
}

async fn get_job_status(db: &PgPool, job_id: Uuid) -> String {
    let row: (String,) =
        sqlx::query_as("SELECT status FROM pipeline_jobs WHERE id = $1")
            .bind(job_id)
            .fetch_one(db)
            .await
            .unwrap();
    row.0
}

async fn get_job_tried(db: &PgPool, job_id: Uuid) -> i32 {
    let row: (i32,) =
        sqlx::query_as("SELECT tried FROM pipeline_jobs WHERE id = $1")
            .bind(job_id)
            .fetch_one(db)
            .await
            .unwrap();
    row.0
}

async fn cleanup(db: &PgPool, job_id: Uuid) {
    sqlx::query("DELETE FROM pipeline_events WHERE job_id = $1")
        .bind(job_id)
        .execute(db)
        .await
        .unwrap();
    sqlx::query("DELETE FROM pipeline_jobs WHERE id = $1")
        .bind(job_id)
        .execute(db)
        .await
        .unwrap();
}

// ── Tests ───────────────────────────────────────────────────────────

/// Success(Next) → advance to next step.
#[tokio::test]
#[ignore]
async fn test_handle_execution_result_success_next() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();
    let job = insert_running_job(&db, job_id, 0, 3).await;

    let next = MockTask::Step2(MockStep {
        value: "advanced".into(),
    });
    let result = ExecutionResult::Success(StepResult::Next(next), serde_json::Value::Null);

    handle_execution_result::<MockTask>(&db, &job, result, &config)
        .await
        .unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "ready", "advanced job should be ready");

    let row: (String,) =
        sqlx::query_as("SELECT current_step FROM pipeline_jobs WHERE id = $1")
            .bind(job_id)
            .fetch_one(&db)
            .await
            .unwrap();
    assert_eq!(row.0, "Step2", "current_step should be Step2");

    cleanup(&db, job_id).await;
}

/// Done → job completed.
#[tokio::test]
#[ignore]
async fn test_handle_execution_result_done() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();
    let job = insert_running_job(&db, job_id, 0, 0).await;

    let result: ExecutionResult<MockTask> =
        ExecutionResult::Done(serde_json::Value::Null);

    handle_execution_result::<MockTask>(&db, &job, result, &config)
        .await
        .unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "completed");

    cleanup(&db, job_id).await;
}

/// Failed with retries remaining → retry.
#[tokio::test]
#[ignore]
async fn test_handle_execution_result_failed_with_retry() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();
    let job = insert_running_job(&db, job_id, 0, 3).await;

    let result: ExecutionResult<MockTask> =
        ExecutionResult::Failed("test error".to_string());

    handle_execution_result::<MockTask>(&db, &job, result, &config)
        .await
        .unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "ready", "retriable failure should reset to ready");

    let tried = get_job_tried(&db, job_id).await;
    assert_eq!(tried, 1, "tried should be incremented to 1");

    cleanup(&db, job_id).await;
}

/// Failed with retries exhausted → permanent failure.
#[tokio::test]
#[ignore]
async fn test_handle_execution_result_failed_exhausted() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();
    let job = insert_running_job(&db, job_id, 1, 1).await;

    let result: ExecutionResult<MockTask> =
        ExecutionResult::Failed("exhausted error".to_string());

    handle_execution_result::<MockTask>(&db, &job, result, &config)
        .await
        .unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "failed", "exhausted failure should be failed");

    cleanup(&db, job_id).await;
}

/// Cancelled → status failed with cancel message.
#[tokio::test]
#[ignore]
async fn test_handle_execution_result_cancelled() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();
    let job = insert_running_job(&db, job_id, 0, 0).await;

    let result: ExecutionResult<MockTask> = ExecutionResult::Cancelled;

    handle_execution_result::<MockTask>(&db, &job, result, &config)
        .await
        .unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "failed", "cancelled should transition to failed");

    cleanup(&db, job_id).await;
}

/// Timeout with retries remaining → retry.
#[tokio::test]
#[ignore]
async fn test_handle_execution_result_timeout() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();
    let job = insert_running_job(&db, job_id, 0, 3).await;

    let result: ExecutionResult<MockTask> = ExecutionResult::Timeout;

    handle_execution_result::<MockTask>(&db, &job, result, &config)
        .await
        .unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "ready", "timeout with retries should reset to ready");

    let tried = get_job_tried(&db, job_id).await;
    assert_eq!(tried, 1, "tried should be incremented to 1");

    cleanup(&db, job_id).await;
}
