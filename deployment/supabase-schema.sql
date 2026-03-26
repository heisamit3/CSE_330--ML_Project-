-- Supabase Schema for Teacher Dashboard
-- Run this in Supabase SQL Editor:
-- https://supabase.com/dashboard/project/YOUR_PROJECT/sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =========================================================
-- 1. User profiles (extends auth.users)
-- =========================================================
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    display_name TEXT,
    role TEXT NOT NULL DEFAULT 'student' CHECK (role IN ('student', 'teacher', 'admin')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================================================
-- 2. Student submissions with AI predictions
-- =========================================================
CREATE TABLE IF NOT EXISTS public.submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    question_id TEXT NOT NULL,
    question_text TEXT NOT NULL,
    selected_answer TEXT NOT NULL,
    student_explanation TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    predicted_label TEXT,
    predicted_category TEXT,
    predicted_misconception TEXT,
    ai_predictions JSONB NOT NULL DEFAULT '[]',
    top_misconception TEXT,
    confidence FLOAT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Migration-safe additions for existing deployed projects
ALTER TABLE
    public.submissions
ADD
    COLUMN IF NOT EXISTS predicted_label TEXT,
ADD
    COLUMN IF NOT EXISTS predicted_category TEXT,
ADD
    COLUMN IF NOT EXISTS predicted_misconception TEXT;

-- =========================================================
-- 3. Teacher feedback with prediction rankings
-- =========================================================
CREATE TABLE IF NOT EXISTS public.teacher_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL REFERENCES public.submissions(id) ON DELETE CASCADE,
    teacher_id UUID NOT NULL REFERENCES public.profiles(id),
    prediction_rankings JSONB NOT NULL DEFAULT '{}',
    additional_prediction TEXT,
    additional_prediction_rank INTEGER CHECK (
        additional_prediction_rank BETWEEN 1
        AND 4
    ),
    feedback_text TEXT NOT NULL,
    correct_prediction_label TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (submission_id, teacher_id)
);

-- =========================================================
-- 4. Student notifications
-- =========================================================
CREATE TABLE IF NOT EXISTS public.notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    submission_id UUID REFERENCES public.submissions(id) ON DELETE CASCADE,
    feedback_id UUID REFERENCES public.teacher_feedback(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================================================
-- Row Level Security
-- =========================================================
ALTER TABLE
    public.profiles ENABLE ROW LEVEL SECURITY;

ALTER TABLE
    public.submissions ENABLE ROW LEVEL SECURITY;

ALTER TABLE
    public.teacher_feedback ENABLE ROW LEVEL SECURITY;

ALTER TABLE
    public.notifications ENABLE ROW LEVEL SECURITY;

CREATE
OR REPLACE FUNCTION public.is_teacher() RETURNS BOOLEAN LANGUAGE sql STABLE AS $ $
SELECT
    COALESCE(
        auth.jwt() -> 'user_metadata' ->> 'role',
        auth.jwt() -> 'app_metadata' ->> 'role',
        ''
    ) IN ('teacher', 'admin');

$ $;

-- =========================================================
-- Profiles policies
-- =========================================================
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;

CREATE POLICY "Users can view own profile" ON public.profiles FOR
SELECT
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Teachers can view all profiles" ON public.profiles;

CREATE POLICY "Teachers can view all profiles" ON public.profiles FOR
SELECT
    USING (public.is_teacher());

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;

CREATE POLICY "Users can update own profile" ON public.profiles FOR
UPDATE
    USING (auth.uid() = id) WITH CHECK (auth.uid() = id);

DROP POLICY IF EXISTS "Service role can insert profiles" ON public.profiles;

CREATE POLICY "Service role can insert profiles" ON public.profiles FOR
INSERT
    WITH CHECK (
        auth.role() = 'service_role'
        OR auth.uid() = id
    );

-- =========================================================
-- Submissions policies
-- =========================================================
DROP POLICY IF EXISTS "Students see own submissions" ON public.submissions;

CREATE POLICY "Students see own submissions" ON public.submissions FOR
SELECT
    USING (student_id = auth.uid());

DROP POLICY IF EXISTS "Students can insert own submissions" ON public.submissions;

CREATE POLICY "Students can insert own submissions" ON public.submissions FOR
INSERT
    WITH CHECK (student_id = auth.uid());

DROP POLICY IF EXISTS "Teachers see all submissions" ON public.submissions;

CREATE POLICY "Teachers see all submissions" ON public.submissions FOR
SELECT
    USING (public.is_teacher());

-- =========================================================
-- Teacher feedback policies
-- =========================================================
DROP POLICY IF EXISTS "Anyone can view feedback" ON public.teacher_feedback;

DROP POLICY IF EXISTS "Students can view feedback on own submissions" ON public.teacher_feedback;

CREATE POLICY "Students can view feedback on own submissions" ON public.teacher_feedback FOR
SELECT
    USING (
        public.is_teacher()
        OR teacher_id = auth.uid()
        OR EXISTS (
            SELECT
                1
            FROM
                public.submissions s
            WHERE
                s.id = submission_id
                AND s.student_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Teachers can insert feedback" ON public.teacher_feedback;

CREATE POLICY "Teachers can insert feedback" ON public.teacher_feedback FOR
INSERT
    WITH CHECK (public.is_teacher());

DROP POLICY IF EXISTS "Teachers can update own feedback" ON public.teacher_feedback;

CREATE POLICY "Teachers can update own feedback" ON public.teacher_feedback FOR
UPDATE
    USING (teacher_id = auth.uid()) WITH CHECK (teacher_id = auth.uid());

-- =========================================================
-- Notifications policies
-- =========================================================
DROP POLICY IF EXISTS "Users see own notifications" ON public.notifications;

CREATE POLICY "Users see own notifications" ON public.notifications FOR
SELECT
    USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Users can update own notifications" ON public.notifications;

CREATE POLICY "Users can update own notifications" ON public.notifications FOR
UPDATE
    USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Service can insert notifications" ON public.notifications;

CREATE POLICY "Service can insert notifications" ON public.notifications FOR
INSERT
    WITH CHECK (
        auth.role() = 'service_role'
        OR public.is_teacher()
    );

-- =========================================================
-- Auto-create profile on user signup
-- =========================================================
CREATE
OR REPLACE FUNCTION public.handle_new_user() RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER AS $ $ BEGIN
INSERT INTO
    public.profiles (id, email, display_name, role)
VALUES
    (
        NEW.id,
        NEW.email,
        COALESCE(
            NEW.raw_user_meta_data ->> 'display_name',
            split_part(NEW.email, '@', 1)
        ),
        COALESCE(NEW.raw_user_meta_data ->> 'role', 'student')
    );

RETURN NEW;

END;

$ $;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

CREATE TRIGGER on_auth_user_created
AFTER
INSERT
    ON auth.users FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =========================================================
-- Indexes for performance
-- =========================================================
CREATE INDEX IF NOT EXISTS idx_submissions_student_id ON public.submissions(student_id);

CREATE INDEX IF NOT EXISTS idx_submissions_created_at ON public.submissions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_teacher_feedback_submission_id ON public.teacher_feedback(submission_id);

CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON public.notifications(user_id);

CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON public.notifications(user_id, is_read);

-- =========================================================
-- Retraining dataset export
-- =========================================================
CREATE
OR REPLACE VIEW public.retraining_dataset AS
SELECT
    s.question_id AS "QuestionId",
    s.question_text AS "QuestionText",
    s.selected_answer AS "MC_answer",
    s.student_explanation AS "StudentExplanation",
    COALESCE(
        s.predicted_category,
        NULLIF(
            split_part(
                COALESCE(
                    s.predicted_label,
                    s.ai_predictions -> 0 ->> 'label',
                    ''
                ),
                ':',
                1
            ),
            ''
        )
    ) AS "CategoryPredicted",
    NULLIF(
        split_part(
            COALESCE(tf.correct_prediction_label, ''),
            ':',
            1
        ),
        ''
    ) AS "CategoryRanked",
    COALESCE(
        NULLIF(
            split_part(
                COALESCE(tf.correct_prediction_label, ''),
                ':',
                2
            ),
            ''
        ),
        NULLIF(s.predicted_misconception, ''),
        NULLIF(s.top_misconception, '')
    ) AS "Misconception"
FROM
    public.submissions s
    LEFT JOIN public.teacher_feedback tf ON tf.submission_id = s.id;

-- =========================================================
-- 5. Teacher-created questions for student practice
-- =========================================================
CREATE TABLE IF NOT EXISTS public.teacher_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    teacher_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    option_a TEXT NOT NULL,
    option_b TEXT NOT NULL,
    option_c TEXT,
    option_d TEXT,
    correct_answer TEXT NOT NULL CHECK (correct_answer IN ('A', 'B', 'C', 'D')),
    -- Expected misconception labels (from the 65 trained labels)
    -- Format: "Misconception_name" without True/False prefix
    expected_misconceptions JSONB NOT NULL DEFAULT '[]',
    -- Category: what subject/topic area
    category TEXT,
    -- Difficulty level
    difficulty TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),
    -- Whether this question has an associated image
    has_image BOOLEAN DEFAULT FALSE,
    image_url TEXT,
    -- Status for workflow
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'archived')),
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for efficient queries
CREATE INDEX IF NOT EXISTS idx_teacher_questions_teacher_id ON public.teacher_questions(teacher_id);

CREATE INDEX IF NOT EXISTS idx_teacher_questions_status ON public.teacher_questions(status);

CREATE INDEX IF NOT EXISTS idx_teacher_questions_created_at ON public.teacher_questions(created_at DESC);

-- Enable RLS
ALTER TABLE
    public.teacher_questions ENABLE ROW LEVEL SECURITY;

-- =========================================================
-- Teacher questions policies
-- =========================================================
-- Teachers can view all questions
DROP POLICY IF EXISTS "Teachers can view all questions" ON public.teacher_questions;

CREATE POLICY "Teachers can view all questions" ON public.teacher_questions FOR
SELECT
    USING (public.is_teacher());

-- Teachers can create questions
DROP POLICY IF EXISTS "Teachers can create questions" ON public.teacher_questions;

CREATE POLICY "Teachers can create questions" ON public.teacher_questions FOR
INSERT
    WITH CHECK (
        public.is_teacher()
        AND teacher_id = auth.uid()
    );

-- Teachers can update their own questions
DROP POLICY IF EXISTS "Teachers can update own questions" ON public.teacher_questions;

CREATE POLICY "Teachers can update own questions" ON public.teacher_questions FOR
UPDATE
    USING (teacher_id = auth.uid()) WITH CHECK (teacher_id = auth.uid());

-- Teachers can delete their own questions
DROP POLICY IF EXISTS "Teachers can delete own questions" ON public.teacher_questions;

CREATE POLICY "Teachers can delete own questions" ON public.teacher_questions FOR DELETE USING (teacher_id = auth.uid());

-- Students can view active questions only
DROP POLICY IF EXISTS "Students can view active questions" ON public.teacher_questions;

CREATE POLICY "Students can view active questions" ON public.teacher_questions FOR
SELECT
    USING (status = 'active');

-- =========================================================
-- Retraining export view for teacher-created questions
-- =========================================================
CREATE
OR REPLACE VIEW public.teacher_questions_export AS
SELECT
    tq.id :: TEXT AS "QuestionId",
    tq.question_text AS "QuestionText",
    CASE
        tq.correct_answer
        WHEN 'A' THEN tq.option_a
        WHEN 'B' THEN tq.option_b
        WHEN 'C' THEN tq.option_c
        WHEN 'D' THEN tq.option_d
    END AS "CorrectAnswer",
    tq.option_a AS "OptionA",
    tq.option_b AS "OptionB",
    tq.option_c AS "OptionC",
    tq.option_d AS "OptionD",
    tq.expected_misconceptions AS "ExpectedMisconceptions",
    tq.category AS "Category",
    tq.difficulty AS "Difficulty",
    p.display_name AS "CreatedBy",
    tq.created_at AS "CreatedAt"
FROM
    public.teacher_questions tq
    JOIN public.profiles p ON p.id = tq.teacher_id
WHERE
    tq.status = 'active';

-- =========================================================
-- 6. Link submissions to teacher questions
-- =========================================================
-- Add column to track question source (original CSV vs teacher-created)
ALTER TABLE public.submissions
ADD COLUMN IF NOT EXISTS question_source TEXT DEFAULT 'original'
    CHECK (question_source IN ('original', 'teacher'));

-- Add column to link to teacher_questions if applicable
ALTER TABLE public.submissions
ADD COLUMN IF NOT EXISTS teacher_question_id UUID REFERENCES public.teacher_questions(id);

-- Index for teacher question lookups
CREATE INDEX IF NOT EXISTS idx_submissions_teacher_question_id
    ON public.submissions(teacher_question_id)
    WHERE teacher_question_id IS NOT NULL;

-- =========================================================
-- 7. Enhanced retraining export (matches train.csv format)
-- =========================================================
-- This view exports data ready for model retraining
-- Format matches: row_id, QuestionId, QuestionText, MC_Answer, StudentExplanation, Category, Misconception
CREATE OR REPLACE VIEW public.retraining_export AS
SELECT
    -- Generate row_id
    ROW_NUMBER() OVER (ORDER BY s.created_at) AS "row_id",

    -- Question ID (use teacher_question_id if from teacher, else original question_id)
    COALESCE(s.teacher_question_id::TEXT, s.question_id) AS "QuestionId",

    -- Question text
    s.question_text AS "QuestionText",

    -- The answer the student selected
    s.selected_answer AS "MC_Answer",

    -- Student's explanation
    s.student_explanation AS "StudentExplanation",

    -- Category: Teacher's corrected label takes priority, then AI prediction
    -- Format: True_Correct, True_Misconception, True_Neither, False_Correct, False_Misconception, False_Neither
    COALESCE(
        NULLIF(split_part(COALESCE(tf.correct_prediction_label, ''), ':', 1), ''),
        s.predicted_category,
        CASE
            WHEN s.is_correct THEN 'True_Correct'
            ELSE 'False_Misconception'
        END
    ) AS "Category",

    -- Misconception: Teacher's correction takes priority
    COALESCE(
        NULLIF(split_part(COALESCE(tf.correct_prediction_label, ''), ':', 2), ''),
        s.predicted_misconception,
        s.top_misconception,
        CASE WHEN s.is_correct THEN '' ELSE 'NA' END
    ) AS "Misconception",

    -- Metadata for tracking
    s.question_source AS "QuestionSource",
    s.is_correct AS "IsCorrect",
    s.confidence AS "AIConfidence",
    tf.id IS NOT NULL AS "TeacherReviewed",
    s.created_at AS "SubmittedAt"

FROM public.submissions s
LEFT JOIN public.teacher_feedback tf ON tf.submission_id = s.id
-- Only export submissions that have been teacher-reviewed for quality data
WHERE tf.id IS NOT NULL;

-- =========================================================
-- 8. View for counting new training samples (for CI/CD trigger)
-- =========================================================
CREATE OR REPLACE FUNCTION public.count_new_training_samples(since_date TIMESTAMPTZ DEFAULT NULL)
RETURNS INTEGER
LANGUAGE sql
STABLE
AS $$
    SELECT COUNT(*)::INTEGER
    FROM public.submissions s
    INNER JOIN public.teacher_feedback tf ON tf.submission_id = s.id
    WHERE since_date IS NULL OR tf.created_at > since_date;
$$;

-- =========================================================
-- 9. Table to track model versions and retraining history
-- =========================================================
CREATE TABLE IF NOT EXISTS public.model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_name TEXT NOT NULL,
    trained_at TIMESTAMPTZ DEFAULT NOW(),
    training_samples_count INTEGER NOT NULL,
    -- Store which submissions were used for this training
    submission_ids JSONB NOT NULL DEFAULT '[]',
    -- Model metrics
    accuracy FLOAT,
    notes TEXT,
    -- Model artifact location
    model_path TEXT,
    created_by UUID REFERENCES public.profiles(id)
);

-- Track when data was last exported for training
ALTER TABLE public.submissions
ADD COLUMN IF NOT EXISTS used_for_training_version UUID REFERENCES public.model_versions(id);