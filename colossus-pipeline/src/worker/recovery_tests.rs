use super::*;
use crate::schema::JobStatus;

use uuid::Uuid;

// ── Helpers ─────────────────────────────────────────────────────────

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

async fn test_db() -> PgPool {
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set for this test");
    PgPool::connect(&url).await.unwrap()
}

/// Insert a running job with configurable heartbeat and timeout fields.
async fn insert_test_job(
    db: &PgPool,
    job_id: Uuid,
    heartbeat_age_secs: Option<i64>,
    timeout_in_past_secs: Option<i64>,
    tried: i32,
    max_retries: i32,
) {
    let heartbeat_expr = match heartbeat_age_secs {
        Some(secs) => format!("NOW() - INTERVAL '{secs} seconds'"),
        None => "NOW()".to_string(),
    };
    let timeout_expr = match timeout_in_past_secs {
        Some(secs) => format!("NOW() - INTERVAL '{secs} seconds'"),
        None => "NULL".to_string(),
    };

    let sql = format!(
        "INSERT INTO pipeline_jobs (id, job_type, job_key, pipeline_version, \
         status, control, current_step, step_data, result, tried, max_retries, \
         retry_delay_secs, priority, wakeup_at, last_heartbeat_at, timeout_at, \
         created_at, updated_at) \
         VALUES ($1, 'test', $2, 1, $3, 'none', 'TestStep', '{{}}', '{{}}', \
         {tried}, {max_retries}, 60, 0, NOW(), {heartbeat_expr}, {timeout_expr}, \
         NOW(), NOW())"
    );

    sqlx::query(&sql)
        .bind(job_id)
        .bind(format!("recovery-test-{job_id}"))
        .bind(JobStatus::Running.as_str())
        .execute(db)
        .await
        .unwrap();
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

#[test]
fn test_recovery_module_compiles() {
    // Ensures recovery.rs compiles and is linked into the test binary.
    // The real verification is the ignored integration tests below.
    assert!(true);
}

/// Empty table — recover_orphans should succeed with no work to do.
#[tokio::test]
#[ignore]
async fn test_recover_orphans_no_zombies_no_timeouts() {
    let db = test_db().await;
    let config = test_config();
    let result = recover_orphans(&db, &config).await;
    assert!(result.is_ok(), "recover_orphans should succeed on empty table");
}

/// Stale heartbeat with retries remaining → reset to ready.
#[tokio::test]
#[ignore]
async fn test_recover_zombie_resets_to_ready() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();

    // Heartbeat 200s old, threshold is 120s → zombie
    insert_test_job(&db, job_id, Some(200), None, 0, 3).await;

    recover_orphans(&db, &config).await.unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "ready", "zombie should be reset to ready");

    cleanup(&db, job_id).await;
}

/// Stale heartbeat with retries exhausted → fail.
#[tokio::test]
#[ignore]
async fn test_recover_zombie_exhausted_fails() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();

    // tried=3, max_retries=3 → exhausted
    insert_test_job(&db, job_id, Some(200), None, 3, 3).await;

    recover_orphans(&db, &config).await.unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "failed", "exhausted zombie should be failed");

    cleanup(&db, job_id).await;
}

/// Timeout in the past with retries remaining → reset to ready.
#[tokio::test]
#[ignore]
async fn test_recover_timeout_resets_to_ready() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();

    // timeout_at 60s in the past, fresh heartbeat
    insert_test_job(&db, job_id, None, Some(60), 0, 3).await;

    recover_orphans(&db, &config).await.unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "ready", "timed-out job should be reset to ready");

    cleanup(&db, job_id).await;
}

/// Timeout in the past with retries exhausted → fail.
#[tokio::test]
#[ignore]
async fn test_recover_timeout_exhausted_fails() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();

    // tried=3, max_retries=3 → exhausted
    insert_test_job(&db, job_id, None, Some(60), 3, 3).await;

    recover_orphans(&db, &config).await.unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "failed", "exhausted timeout should be failed");

    cleanup(&db, job_id).await;
}

/// Fresh heartbeat should not be affected by recovery.
#[tokio::test]
#[ignore]
async fn test_fresh_heartbeat_not_affected() {
    let db = test_db().await;
    let config = test_config();
    let job_id = Uuid::new_v4();

    // Fresh heartbeat (0s old), no timeout
    insert_test_job(&db, job_id, None, None, 0, 3).await;

    recover_orphans(&db, &config).await.unwrap();

    let status = get_job_status(&db, job_id).await;
    assert_eq!(status, "running", "fresh job should still be running");

    cleanup(&db, job_id).await;
}
