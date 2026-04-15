//! Integration tests for colossus-pipeline.
//!
//! All tests require a live PostgreSQL database at TEST_DATABASE_URL.
//! Run with: cargo test -p colossus-pipeline --test integration -- --ignored --test-threads=1 --nocapture

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use colossus_pipeline::cancel::CancellationToken;
use colossus_pipeline::progress::ProgressReporter;
use colossus_pipeline::worker::config::WorkerConfig;
use colossus_pipeline::{
    JobRow, JobStatus, PipelineError, PipelineEvent, Scheduler, StepResult, Task,
    Worker,
};

// ═══════════════════════════════════════════════════════════════════
// Test infrastructure
// ═══════════════════════════════════════════════════════════════════

async fn test_pool() -> PgPool {
    let url =
        std::env::var("TEST_DATABASE_URL").expect("TEST_DATABASE_URL must be set for integration tests");
    PgPool::connect(&url)
        .await
        .expect("Failed to connect to test database")
}

async fn setup_schema(db: &PgPool) {
    sqlx::query("DROP TABLE IF EXISTS pipeline_events CASCADE")
        .execute(db)
        .await
        .unwrap();
    sqlx::query("DROP TABLE IF EXISTS pipeline_jobs CASCADE")
        .execute(db)
        .await
        .unwrap();
    sqlx::query("DROP FUNCTION IF EXISTS pipeline_jobs_notify() CASCADE")
        .execute(db)
        .await
        .unwrap();

    let migration_001 = include_str!("../migrations/001_create_pipeline_jobs.sql");
    sqlx::raw_sql(migration_001).execute(db).await.unwrap();

    let migration_002 = include_str!("../migrations/002_create_pipeline_events.sql");
    sqlx::raw_sql(migration_002).execute(db).await.unwrap();
}

async fn cleanup(db: &PgPool) {
    sqlx::query("DELETE FROM pipeline_events")
        .execute(db)
        .await
        .ok();
    sqlx::query("DELETE FROM pipeline_jobs")
        .execute(db)
        .await
        .ok();
}

// ═══════════════════════════════════════════════════════════════════
// TestTask — 3-step task with injectable behavior
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TestTask {
    StepA { value: String },
    StepB { value: String },
    StepC { value: String },
}

struct TestContext {
    fail_step_b: AtomicBool,
    slow_step_b: AtomicBool,
    slow_step_b_secs: AtomicU64,
    fail_delete: AtomicBool,
}

impl TestContext {
    fn new() -> Self {
        Self {
            fail_step_b: AtomicBool::new(false),
            slow_step_b: AtomicBool::new(false),
            slow_step_b_secs: AtomicU64::new(10),
            fail_delete: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl Task for TestTask {
    type Context = TestContext;

    fn current_step_name(&self) -> &'static str {
        match self {
            TestTask::StepA { .. } => "StepA",
            TestTask::StepB { .. } => "StepB",
            TestTask::StepC { .. } => "StepC",
        }
    }

    fn validate_transition(current: &str, next: &str) -> Result<(), PipelineError> {
        match (current, next) {
            ("StepA", "StepB") | ("StepB", "StepC") => Ok(()),
            _ => Err(PipelineError::InvalidTransition {
                from: current.to_string(),
                to: next.to_string(),
            }),
        }
    }

    async fn execute_current(
        self,
        _db: &PgPool,
        context: &TestContext,
        _cancel: &CancellationToken,
        _progress: &ProgressReporter,
    ) -> Result<StepResult<Self>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            TestTask::StepA { value } => Ok(StepResult::Next(TestTask::StepB { value })),
            TestTask::StepB { value } => {
                if context.slow_step_b.load(Ordering::SeqCst) {
                    let secs = context.slow_step_b_secs.load(Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_secs(secs)).await;
                }
                if context.fail_step_b.load(Ordering::SeqCst) {
                    return Err("StepB intentional failure".into());
                }
                Ok(StepResult::Next(TestTask::StepC { value }))
            }
            TestTask::StepC { .. } => Ok(StepResult::Done),
        }
    }

    async fn on_cancel_current(
        self,
        _db: &PgPool,
        _context: &TestContext,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    async fn on_delete_current(
        self,
        _db: &PgPool,
        context: &TestContext,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if context.fail_delete.load(Ordering::SeqCst) {
            return Err("on_delete intentional failure".into());
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Worker helpers
// ═══════════════════════════════════════════════════════════════════

fn test_worker_config() -> WorkerConfig {
    WorkerConfig {
        max_concurrent: 2,
        poll_interval_secs: 1,
        drain_timeout_secs: 5,
        heartbeat_interval_secs: 2,
        zombie_threshold_secs: 5,
        recovery_interval_secs: 3,
        recovery_wakeup_secs: 1,
        cancel_poll_interval_secs: 1,
        notify_channel: "pipeline_jobs_changed".to_string(),
        worker_id: Uuid::new_v4().to_string(),
    }
}

async fn spawn_worker(
    db: PgPool,
    context: Arc<TestContext>,
) -> (tokio::task::JoinHandle<()>, tokio::sync::watch::Sender<bool>) {
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let config = test_worker_config();
    let worker = Worker::<TestTask>::new(db, context, config, shutdown_rx);
    let handle = tokio::spawn(async move {
        if let Err(e) = worker.run().await {
            eprintln!("Worker exited with error: {e}");
        }
    });
    // Give worker time to start and run initial recovery
    tokio::time::sleep(Duration::from_millis(500)).await;
    (handle, shutdown_tx)
}

async fn shutdown_worker(
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    handle: tokio::task::JoinHandle<()>,
) {
    let _ = shutdown_tx.send(true);
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .ok();
}

// ═══════════════════════════════════════════════════════════════════
// Polling helpers
// ═══════════════════════════════════════════════════════════════════

async fn wait_for_status(
    scheduler: &Scheduler<'_>,
    job_id: Uuid,
    target: JobStatus,
    timeout_secs: u64,
) -> JobRow {
    let start = Instant::now();
    loop {
        let job = scheduler.status(job_id).await.unwrap();
        if job.status == target {
            return job;
        }
        if start.elapsed() > Duration::from_secs(timeout_secs) {
            panic!(
                "Timeout waiting for status {:?}, current: {:?}",
                target, job.status
            );
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn wait_for_terminal(
    scheduler: &Scheduler<'_>,
    job_id: Uuid,
    timeout_secs: u64,
) -> JobRow {
    let start = Instant::now();
    loop {
        let job = scheduler.status(job_id).await.unwrap();
        match job.status {
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled => return job,
            _ => {}
        }
        if start.elapsed() > Duration::from_secs(timeout_secs) {
            panic!(
                "Timeout waiting for terminal status, current: {:?}",
                job.status
            );
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

fn has_event(events: &[PipelineEvent], event_type: &str) -> bool {
    events.iter().any(|e| e.event_type == event_type)
}

fn task_a(value: &str) -> TestTask {
    TestTask::StepA {
        value: value.to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[tokio::test]
#[ignore]
async fn t01_happy_path_submit_to_completed() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    let s = Scheduler::new(&db);

    let job_id = s
        .submit::<TestTask>("test", "t01", task_a("hello"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;

    let job = wait_for_status(&s, job_id, JobStatus::Completed, 5).await;
    assert_eq!(job.status, JobStatus::Completed);

    let events = s.events(job_id).await.unwrap();
    assert!(has_event(&events, "job_submitted"));
    assert!(has_event(&events, "step_started"));
    assert!(has_event(&events, "step_completed"));
    assert!(has_event(&events, "job_completed"));

    shutdown_worker(tx, handle).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t02_submit_returns_valid_uuid() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("mytype", "mykey", task_a("v"), 5, Some("test-user"))
        .await
        .unwrap();

    assert!(!job_id.is_nil());

    let job = s.status(job_id).await.unwrap();
    assert_eq!(job.job_type, "mytype");
    assert_eq!(job.job_key, "mykey");
    assert_eq!(job.current_step, "StepA");
    assert_eq!(job.priority, 5);

    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t03_duplicate_job_rejected() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let s = Scheduler::new(&db);
    s.submit::<TestTask>("test", "dup", task_a("a"), 0, None)
        .await
        .unwrap();

    let err = s
        .submit::<TestTask>("test", "dup", task_a("b"), 0, None)
        .await
        .unwrap_err();

    assert!(
        matches!(err, PipelineError::DuplicateJob { .. }),
        "expected DuplicateJob, got {err:?}"
    );

    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t04_submit_allowed_after_completed() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    let s = Scheduler::new(&db);

    let id1 = s
        .submit::<TestTask>("test", "resubmit", task_a("first"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;
    wait_for_status(&s, id1, JobStatus::Completed, 5).await;
    shutdown_worker(tx, handle).await;

    let id2 = s
        .submit::<TestTask>("test", "resubmit", task_a("second"), 0, None)
        .await
        .unwrap();

    assert_ne!(id1, id2);

    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t05_cancel_running_job() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    ctx.slow_step_b.store(true, Ordering::SeqCst);

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("test", "cancel", task_a("c"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;

    // Wait until job reaches Running (StepB is sleeping 10s)
    wait_for_status(&s, job_id, JobStatus::Running, 5).await;

    s.cancel(job_id).await.unwrap();

    let job = wait_for_terminal(&s, job_id, 10).await;
    assert_eq!(job.status, JobStatus::Failed, "cancelled job should be Failed");

    let events = s.events(job_id).await.unwrap();
    assert!(
        has_event(&events, "cancelled") || has_event(&events, "cancel_requested"),
        "expected cancel event in log"
    );

    shutdown_worker(tx, handle).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t06_cancel_completed_job_rejected() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    let s = Scheduler::new(&db);

    let job_id = s
        .submit::<TestTask>("test", "cancel-done", task_a("d"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;
    wait_for_status(&s, job_id, JobStatus::Completed, 5).await;
    shutdown_worker(tx, handle).await;

    let err = s.cancel(job_id).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::JobNotCancellable(_)),
        "expected JobNotCancellable, got {err:?}"
    );

    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t07_retry_on_failure() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    ctx.fail_step_b.store(true, Ordering::SeqCst);

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("test", "retry", task_a("r"), 0, None)
        .await
        .unwrap();

    // Set max_retries so the job retries instead of failing immediately
    sqlx::query("UPDATE pipeline_jobs SET max_retries = $1, retry_delay_secs = $2 WHERE id = $3")
        .bind(2)
        .bind(1)
        .bind(job_id)
        .execute(&db)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;

    // Wait for at least one retry attempt
    tokio::time::sleep(Duration::from_secs(5)).await;

    let job = s.status(job_id).await.unwrap();
    assert!(job.tried > 0, "tried should be > 0 after retry, got {}", job.tried);

    let events = s.events(job_id).await.unwrap();
    assert!(has_event(&events, "step_failed"));
    assert!(has_event(&events, "retry_scheduled"));

    shutdown_worker(tx, handle).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t08_retry_exhausted() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    ctx.fail_step_b.store(true, Ordering::SeqCst);

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("test", "exhaust", task_a("e"), 0, None)
        .await
        .unwrap();

    sqlx::query("UPDATE pipeline_jobs SET max_retries = $1, retry_delay_secs = $2 WHERE id = $3")
        .bind(1)
        .bind(1)
        .bind(job_id)
        .execute(&db)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;

    let job = wait_for_status(&s, job_id, JobStatus::Failed, 10).await;
    assert_eq!(job.status, JobStatus::Failed);

    let events = s.events(job_id).await.unwrap();
    assert!(has_event(&events, "step_exhausted"));

    shutdown_worker(tx, handle).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t09_timeout_fires() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    ctx.slow_step_b.store(true, Ordering::SeqCst);

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("test", "timeout", task_a("t"), 0, None)
        .await
        .unwrap();

    // Set timeout_at before claim — COALESCE in claim() preserves it
    sqlx::query(
        "UPDATE pipeline_jobs SET timeout_at = NOW() + INTERVAL '3 seconds' WHERE id = $1",
    )
    .bind(job_id)
    .execute(&db)
    .await
    .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;

    let job = wait_for_terminal(&s, job_id, 10).await;
    assert_eq!(job.status, JobStatus::Failed, "timed-out job should be Failed");

    let events = s.events(job_id).await.unwrap();
    let has_timeout_msg = events
        .iter()
        .any(|e| e.event_type == "step_failed" && e.message.contains("timed out"));
    assert!(has_timeout_msg, "expected step_failed with 'timed out' message");

    shutdown_worker(tx, handle).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t10_resume_failed_job() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    ctx.fail_step_b.store(true, Ordering::SeqCst);

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("test", "resume", task_a("res"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), Arc::clone(&ctx)).await;
    wait_for_status(&s, job_id, JobStatus::Failed, 5).await;
    shutdown_worker(tx, handle).await;

    // Clear the failure flag and resume
    ctx.fail_step_b.store(false, Ordering::SeqCst);
    s.resume(job_id).await.unwrap();

    let (handle2, tx2) = spawn_worker(db.clone(), ctx).await;
    let job = wait_for_status(&s, job_id, JobStatus::Completed, 5).await;
    assert_eq!(job.status, JobStatus::Completed);

    shutdown_worker(tx2, handle2).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t11_resume_not_failed_rejected() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("test", "resume-bad", task_a("rb"), 0, None)
        .await
        .unwrap();

    let err = s.resume(job_id).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::JobNotResumable(_)),
        "expected JobNotResumable, got {err:?}"
    );

    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t12_delete_completed_job() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    let s = Scheduler::new(&db);

    let job_id = s
        .submit::<TestTask>("test", "delete", task_a("del"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx.clone()).await;
    wait_for_status(&s, job_id, JobStatus::Completed, 10).await;
    shutdown_worker(tx, handle).await;

    s.delete::<TestTask>(job_id, &ctx).await.unwrap();

    let err = s.status(job_id).await.unwrap_err();
    assert!(matches!(err, PipelineError::NotFound(_)));

    // Events should also be gone (CASCADE)
    let events: Vec<(i64,)> =
        sqlx::query_as("SELECT id FROM pipeline_events WHERE job_id = $1")
            .bind(job_id)
            .fetch_all(&db)
            .await
            .unwrap();
    assert!(events.is_empty(), "events should be cascade-deleted");

    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t13_delete_running_blocked() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    ctx.slow_step_b.store(true, Ordering::SeqCst);

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("test", "del-run", task_a("dr"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx.clone()).await;
    wait_for_status(&s, job_id, JobStatus::Running, 3).await;

    let err = s.delete::<TestTask>(job_id, &ctx).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::JobRunning(_)),
        "expected JobRunning, got {err:?}"
    );

    shutdown_worker(tx, handle).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t14_delete_rollback_on_cleanup_error() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    ctx.fail_delete.store(true, Ordering::SeqCst);

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("test", "del-fail", task_a("df"), 0, None)
        .await
        .unwrap();

    let err = s.delete::<TestTask>(job_id, &ctx).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::Cleanup(_)),
        "expected Cleanup, got {err:?}"
    );

    // Job should still exist
    let job = s.status(job_id).await.unwrap();
    assert_eq!(job.id, job_id);

    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t15_shutdown_drains_jobs() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    // Make StepB slow enough to observe Running, but fast enough to drain
    ctx.slow_step_b.store(true, Ordering::SeqCst);
    ctx.slow_step_b_secs.store(3, Ordering::SeqCst);

    let s = Scheduler::new(&db);

    let job_id = s
        .submit::<TestTask>("test", "drain", task_a("drain"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;

    // Wait for the job to start running (StepB sleeping 3s)
    wait_for_status(&s, job_id, JobStatus::Running, 5).await;

    // Send shutdown — worker should drain in-flight jobs within drain_timeout (5s)
    shutdown_worker(tx, handle).await;

    // Job should have completed during drain
    let job = wait_for_terminal(&s, job_id, 10).await;
    assert_eq!(
        job.status,
        JobStatus::Completed,
        "job should complete during drain"
    );

    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t16_concurrent_max_two() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    ctx.slow_step_b.store(true, Ordering::SeqCst);
    ctx.slow_step_b_secs.store(3, Ordering::SeqCst);

    let s = Scheduler::new(&db);
    for i in 0..3 {
        s.submit::<TestTask>(
            "test",
            &format!("conc-{i}"),
            task_a(&format!("c{i}")),
            0,
            None,
        )
        .await
        .unwrap();
    }

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;

    // Poll until 2 jobs are Running (more robust than a single timed check)
    let start = Instant::now();
    loop {
        let running: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM pipeline_jobs WHERE status = $1",
        )
        .bind(JobStatus::Running.as_str())
        .fetch_one(&db)
        .await
        .unwrap();

        if running.0 == 2 {
            break;
        }
        if start.elapsed() > Duration::from_secs(5) {
            panic!(
                "Timeout waiting for 2 running jobs, got {}",
                running.0
            );
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    shutdown_worker(tx, handle).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t17_zombie_recovery() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    let s = Scheduler::new(&db);

    let job_id = s
        .submit::<TestTask>("test", "zombie", task_a("z"), 0, None)
        .await
        .unwrap();

    // Set max_retries=1 so zombie recovery resets to Ready instead of
    // immediately exhausting (tried=0 >= max_retries=0 would fail the job).
    sqlx::query("UPDATE pipeline_jobs SET max_retries = 1 WHERE id = $1")
        .bind(job_id)
        .execute(&db)
        .await
        .unwrap();

    // Simulate a crashed worker: set Running with stale heartbeat
    sqlx::query(
        "UPDATE pipeline_jobs SET status = $1, worker_id = 'dead-worker', \
         last_heartbeat_at = NOW() - INTERVAL '10 seconds' WHERE id = $2",
    )
    .bind(JobStatus::Running.as_str())
    .bind(job_id)
    .execute(&db)
    .await
    .unwrap();

    // Start worker — recovery runs on startup
    let (handle, tx) = spawn_worker(db.clone(), ctx).await;

    let job = wait_for_status(&s, job_id, JobStatus::Completed, 10).await;
    assert_eq!(job.status, JobStatus::Completed);

    let events = s.events(job_id).await.unwrap();
    assert!(has_event(&events, "zombie_recovered"));

    shutdown_worker(tx, handle).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t18_events_chronological() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    let s = Scheduler::new(&db);

    let job_id = s
        .submit::<TestTask>("test", "events", task_a("ev"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;
    wait_for_status(&s, job_id, JobStatus::Completed, 5).await;
    shutdown_worker(tx, handle).await;

    let events = s.events(job_id).await.unwrap();
    assert!(events.len() >= 2, "expected at least 2 events");
    assert!(has_event(&events, "job_submitted"));
    assert!(has_event(&events, "job_completed"));

    // Verify chronological order
    for pair in events.windows(2) {
        assert!(
            pair[1].created_at >= pair[0].created_at,
            "events out of order: {:?} after {:?}",
            pair[0].event_type,
            pair[1].event_type
        );
    }

    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t19_list_with_filters() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let ctx = Arc::new(TestContext::new());
    let s = Scheduler::new(&db);

    s.submit::<TestTask>("type_a", "la1", task_a("1"), 0, None)
        .await
        .unwrap();
    s.submit::<TestTask>("type_a", "la2", task_a("2"), 0, None)
        .await
        .unwrap();
    s.submit::<TestTask>("type_b", "lb1", task_a("3"), 0, None)
        .await
        .unwrap();

    let (handle, tx) = spawn_worker(db.clone(), ctx).await;

    // Wait for all to complete
    tokio::time::sleep(Duration::from_secs(3)).await;

    let type_a = s.list(Some("type_a"), None, 10, 0).await.unwrap();
    assert_eq!(type_a.len(), 2, "expected 2 type_a jobs");

    let completed = s
        .list(None, Some(JobStatus::Completed.as_str()), 10, 0)
        .await
        .unwrap();
    assert_eq!(completed.len(), 3, "expected 3 completed jobs");

    let type_b_completed = s
        .list(
            Some("type_b"),
            Some(JobStatus::Completed.as_str()),
            10,
            0,
        )
        .await
        .unwrap();
    assert_eq!(type_b_completed.len(), 1, "expected 1 type_b completed job");

    shutdown_worker(tx, handle).await;
    cleanup(&db).await;
}

#[tokio::test]
#[ignore]
async fn t20_validate_transition_rejected() {
    let db = test_pool().await;
    setup_schema(&db).await;

    let s = Scheduler::new(&db);
    let job_id = s
        .submit::<TestTask>("test", "trans", task_a("tr"), 0, None)
        .await
        .unwrap();

    // Try to advance StepA → StepA (invalid)
    let err = s
        .advance::<TestTask>(
            job_id,
            TestTask::StepA {
                value: "bad".to_string(),
            },
        )
        .await
        .unwrap_err();

    assert!(
        matches!(err, PipelineError::InvalidTransition { .. }),
        "expected InvalidTransition, got {err:?}"
    );

    cleanup(&db).await;
}
