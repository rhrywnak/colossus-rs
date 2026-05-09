use super::*;
use crate::cancel::CancellationToken;
use crate::error::PipelineError;
use crate::progress::ProgressReporter;
use crate::step::StepResult;

use async_trait::async_trait;

// ── MockTask for integration tests ──────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MockStep {
    value: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum MockTask {
    Init(MockStep),
}

#[async_trait]
impl Task for MockTask {
    type Context = ();

    fn current_step_name(&self) -> &'static str {
        match self {
            MockTask::Init(_) => "Init",
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

/// MockTask whose on_delete_current returns an error.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum FailDeleteTask {
    Init(MockStep),
}

#[async_trait]
impl Task for FailDeleteTask {
    type Context = ();

    fn current_step_name(&self) -> &'static str {
        "Init"
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
        Err("cleanup failed on purpose".into())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

async fn test_db() -> PgPool {
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set for this test");
    PgPool::connect(&url).await.unwrap()
}

fn mock_task() -> MockTask {
    MockTask::Init(MockStep {
        value: "test".into(),
    })
}

/// Unique job key to avoid collisions between concurrent test runs.
fn unique_key() -> String {
    format!("sched-test-{}", Uuid::new_v4())
}

async fn cleanup(db: &PgPool, job_id: Uuid) {
    // Events cascade-deleted by FK
    sqlx::query("DELETE FROM pipeline_jobs WHERE id = $1")
        .bind(job_id)
        .execute(db)
        .await
        .unwrap();
}

async fn set_status(db: &PgPool, job_id: Uuid, status: &str) {
    sqlx::query("UPDATE pipeline_jobs SET status = $1, updated_at = NOW() WHERE id = $2")
        .bind(status)
        .bind(job_id)
        .execute(db)
        .await
        .unwrap();
}

// ── Tests ───────────────────────────────────────────────────────────

/// Submit a job and verify the returned UUID can be fetched.
#[tokio::test]
#[ignore]
async fn test_submit_returns_uuid() {
    let db = test_db().await;
    let s = Scheduler::new(&db);
    let key = unique_key();

    let job_id = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap();

    let job = s.status(job_id).await.unwrap();
    assert_eq!(job.job_type, "test");
    assert_eq!(job.job_key, key);
    assert_eq!(job.current_step, "Init");

    cleanup(&db, job_id).await;
}

/// Duplicate submit with same job_type+job_key returns DuplicateJob.
#[tokio::test]
#[ignore]
async fn test_submit_duplicate_returns_error() {
    let db = test_db().await;
    let s = Scheduler::new(&db);
    let key = unique_key();

    let job_id = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap();

    let err = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap_err();

    assert!(
        matches!(err, PipelineError::DuplicateJob { .. }),
        "expected DuplicateJob, got {err:?}"
    );

    cleanup(&db, job_id).await;
}

/// Submit after previous job completed — should succeed.
#[tokio::test]
#[ignore]
async fn test_submit_allows_after_completed() {
    let db = test_db().await;
    let s = Scheduler::new(&db);
    let key = unique_key();

    let job_id1 = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap();

    set_status(&db, job_id1, JobStatus::Completed.as_str()).await;

    let job_id2 = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap();

    assert_ne!(job_id1, job_id2);

    cleanup(&db, job_id1).await;
    cleanup(&db, job_id2).await;
}

/// Cancel a completed job returns JobNotCancellable.
#[tokio::test]
#[ignore]
async fn test_cancel_not_cancellable() {
    let db = test_db().await;
    let s = Scheduler::new(&db);
    let key = unique_key();

    let job_id = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap();

    set_status(&db, job_id, JobStatus::Completed.as_str()).await;

    let err = s.cancel(job_id).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::JobNotCancellable(_)),
        "expected JobNotCancellable, got {err:?}"
    );

    cleanup(&db, job_id).await;
}

/// Resume a ready (non-failed, non-parked) job returns JobNotResumable.
#[tokio::test]
#[ignore]
async fn test_resume_not_resumable() {
    let db = test_db().await;
    let s = Scheduler::new(&db);
    let key = unique_key();

    let job_id = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap();

    let err = s.resume(job_id).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::JobNotResumable(_)),
        "expected JobNotResumable, got {err:?}"
    );

    cleanup(&db, job_id).await;
}

/// Delete a running job returns JobRunning.
#[tokio::test]
#[ignore]
async fn test_delete_running_blocked() {
    let db = test_db().await;
    let s = Scheduler::new(&db);
    let key = unique_key();

    let job_id = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap();

    set_status(&db, job_id, JobStatus::Running.as_str()).await;

    let err = s.delete::<MockTask>(job_id, &()).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::JobRunning(_)),
        "expected JobRunning, got {err:?}"
    );

    cleanup(&db, job_id).await;
}

/// Delete a ready job succeeds and removes the row.
#[tokio::test]
#[ignore]
async fn test_delete_calls_on_delete() {
    let db = test_db().await;
    let s = Scheduler::new(&db);
    let key = unique_key();

    let job_id = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap();

    s.delete::<MockTask>(job_id, &()).await.unwrap();

    let err = s.status(job_id).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::NotFound(_)),
        "expected NotFound after delete, got {err:?}"
    );
}

/// Delete with cleanup error preserves the job row.
#[tokio::test]
#[ignore]
async fn test_delete_rollback_on_cleanup_error() {
    let db = test_db().await;
    let s = Scheduler::new(&db);
    let key = unique_key();

    // Submit as MockTask (so the INSERT shape matches)
    let job_id = s
        .submit::<MockTask>("test", &key, mock_task(), 0, None)
        .await
        .unwrap();

    // Overwrite step_data to match FailDeleteTask's enum shape
    let fail_task = FailDeleteTask::Init(MockStep {
        value: "fail".into(),
    });
    let fail_data = serde_json::to_value(&fail_task).unwrap();
    sqlx::query("UPDATE pipeline_jobs SET step_data = $1 WHERE id = $2")
        .bind(&fail_data)
        .bind(job_id)
        .execute(&db)
        .await
        .unwrap();

    let err = s.delete::<FailDeleteTask>(job_id, &()).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::Cleanup(_)),
        "expected Cleanup error, got {err:?}"
    );

    // Job should still exist
    let job = s.status(job_id).await.unwrap();
    assert_eq!(job.id, job_id, "job should still exist after cleanup error");

    cleanup(&db, job_id).await;
}
