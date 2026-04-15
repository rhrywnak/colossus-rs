use super::*;
use crate::cancel::CancellationToken;
use crate::error::PipelineError;
use crate::progress::ProgressReporter;
use crate::step::StepResult;

use async_trait::async_trait;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum MockTask {
    Init,
}

#[async_trait]
impl Task for MockTask {
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
        Ok(())
    }
}

#[tokio::test]
async fn test_worker_construction() {
    let db =
        PgPool::connect_lazy("postgres://invalid:5432/none").expect("connect_lazy should not fail");
    let context = Arc::new(());
    let config = WorkerConfig::from_env();
    let (_tx, rx) = tokio::sync::watch::channel(false);

    let worker = Worker::<MockTask>::new(db, context, config, rx);
    drop(worker);
}
