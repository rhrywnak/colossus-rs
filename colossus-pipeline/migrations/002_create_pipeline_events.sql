-- Pipeline events table.
-- Append-only audit log. Every state transition and notable occurrence
-- is recorded here. ON DELETE CASCADE removes events when the job is deleted.
CREATE TABLE IF NOT EXISTS pipeline_events (
    id          BIGSERIAL   PRIMARY KEY,
    job_id      UUID        NOT NULL REFERENCES pipeline_jobs(id) ON DELETE CASCADE,
    step        TEXT        NOT NULL,
    event_type  TEXT        NOT NULL,
    message     TEXT        NOT NULL,
    details     JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_events_job_timeline
    ON pipeline_events (job_id, created_at ASC);
