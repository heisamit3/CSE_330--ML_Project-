-- Supabase Schema for MAP Student Practice
-- Run this in Supabase SQL Editor for a FRESH database
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
    -- AI model predictions
    predicted_label TEXT,
    predicted_category TEXT,
    predicted_misconception TEXT,
    ai_predictions JSONB NOT NULL DEFAULT '[]',
    top_misconception TEXT,
    confidence FLOAT DEFAULT 0,
    -- Question source tracking
    question_source TEXT DEFAULT 'original' CHECK (question_source IN ('original', 'teacher')),
    teacher_question_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================================================
-- 3. Teacher feedback with prediction rankings
-- =========================================================
CREATE TABLE IF NOT EXISTS public.teacher_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL REFERENCES public.submissions(id) ON DELETE CASCADE,
    teacher_id UUID NOT NULL REFERENCES public.profiles(id),
    prediction_rankings JSONB NOT NULL DEFAULT '{}',
    additional_prediction TEXT,
    additional_prediction_rank INTEGER CHECK (additional_prediction_rank BETWEEN 1 AND 4),
    feedback_text TEXT NOT NULL,
    -- The label the teacher ranked as #1 (correct)
    -- Format: "Category:Misconception" e.g. "False_Misconception:Additive"
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
    expected_misconceptions JSONB NOT NULL DEFAULT '[]',
    category TEXT,
    difficulty TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),
    has_image BOOLEAN DEFAULT FALSE,
    image_url TEXT,
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'archived')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add foreign key for teacher_question_id after teacher_questions table exists
ALTER TABLE public.submissions
ADD CONSTRAINT fk_submissions_teacher_question
FOREIGN KEY (teacher_question_id) REFERENCES public.teacher_questions(id);

-- =========================================================
-- Row Level Security
-- =========================================================
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.teacher_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.teacher_questions ENABLE ROW LEVEL SECURITY;

-- Helper function to check if user is teacher
CREATE OR REPLACE FUNCTION public.is_teacher()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(
        auth.jwt() -> 'user_metadata' ->> 'role',
        auth.jwt() -> 'app_metadata' ->> 'role',
        ''
    ) IN ('teacher', 'admin');
$$;

-- =========================================================
-- Profiles policies
-- =========================================================
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Teachers can view all profiles" ON public.profiles
    FOR SELECT USING (public.is_teacher());

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id) WITH CHECK (auth.uid() = id);

CREATE POLICY "Service role can insert profiles" ON public.profiles
    FOR INSERT WITH CHECK (auth.role() = 'service_role' OR auth.uid() = id);

-- =========================================================
-- Submissions policies
-- =========================================================
CREATE POLICY "Students see own submissions" ON public.submissions
    FOR SELECT USING (student_id = auth.uid());

CREATE POLICY "Students can insert own submissions" ON public.submissions
    FOR INSERT WITH CHECK (student_id = auth.uid());

CREATE POLICY "Teachers see all submissions" ON public.submissions
    FOR SELECT USING (public.is_teacher());

-- =========================================================
-- Teacher feedback policies
-- =========================================================
CREATE POLICY "Students can view feedback on own submissions" ON public.teacher_feedback
    FOR SELECT USING (
        public.is_teacher()
        OR teacher_id = auth.uid()
        OR EXISTS (
            SELECT 1 FROM public.submissions s
            WHERE s.id = submission_id AND s.student_id = auth.uid()
        )
    );

CREATE POLICY "Teachers can insert feedback" ON public.teacher_feedback
    FOR INSERT WITH CHECK (public.is_teacher());

CREATE POLICY "Teachers can update own feedback" ON public.teacher_feedback
    FOR UPDATE USING (teacher_id = auth.uid()) WITH CHECK (teacher_id = auth.uid());

-- =========================================================
-- Notifications policies
-- =========================================================
CREATE POLICY "Users see own notifications" ON public.notifications
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can update own notifications" ON public.notifications
    FOR UPDATE USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());

CREATE POLICY "Service can insert notifications" ON public.notifications
    FOR INSERT WITH CHECK (auth.role() = 'service_role' OR public.is_teacher());

-- =========================================================
-- Teacher questions policies
-- =========================================================
CREATE POLICY "Teachers can view all questions" ON public.teacher_questions
    FOR SELECT USING (public.is_teacher());

CREATE POLICY "Teachers can create questions" ON public.teacher_questions
    FOR INSERT WITH CHECK (public.is_teacher() AND teacher_id = auth.uid());

CREATE POLICY "Teachers can update own questions" ON public.teacher_questions
    FOR UPDATE USING (teacher_id = auth.uid()) WITH CHECK (teacher_id = auth.uid());

CREATE POLICY "Teachers can delete own questions" ON public.teacher_questions
    FOR DELETE USING (teacher_id = auth.uid());

CREATE POLICY "Students can view active questions" ON public.teacher_questions
    FOR SELECT USING (status = 'active');

-- =========================================================
-- Auto-create profile on user signup
-- =========================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    INSERT INTO public.profiles (id, email, display_name, role)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data ->> 'display_name', split_part(NEW.email, '@', 1)),
        COALESCE(NEW.raw_user_meta_data ->> 'role', 'student')
    );
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =========================================================
-- Indexes for performance
-- =========================================================
CREATE INDEX IF NOT EXISTS idx_submissions_student_id ON public.submissions(student_id);
CREATE INDEX IF NOT EXISTS idx_submissions_created_at ON public.submissions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_teacher_question_id ON public.submissions(teacher_question_id) WHERE teacher_question_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_teacher_feedback_submission_id ON public.teacher_feedback(submission_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON public.notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON public.notifications(user_id, is_read);
CREATE INDEX IF NOT EXISTS idx_teacher_questions_teacher_id ON public.teacher_questions(teacher_id);
CREATE INDEX IF NOT EXISTS idx_teacher_questions_status ON public.teacher_questions(status);

-- =========================================================
-- RETRAINING DATASET VIEW (UNRESTRICTED - for ML pipeline)
-- =========================================================
-- This view is accessible without authentication for data export
-- Contains: MisconceptionPredicted (AI) and MisconceptionRanked (Teacher)
--
-- Data flow:
-- 1. Student submits answer → AI predicts misconception → stored in submissions table
-- 2. Teacher reviews and ranks predictions → stored in teacher_feedback table
-- 3. This view joins both to show predicted vs ranked side by side
-- =========================================================
CREATE OR REPLACE VIEW public.retraining_dataset AS
SELECT
    s.id AS "SubmissionId",
    s.question_id AS "QuestionId",
    s.question_text AS "QuestionText",
    s.selected_answer AS "MC_Answer",
    s.student_explanation AS "StudentExplanation",
    s.is_correct AS "IsCorrect",

    -- ===========================================
    -- AI MODEL PREDICTIONS (what the model said)
    -- ===========================================
    COALESCE(
        s.predicted_category,
        NULLIF(split_part(COALESCE(s.predicted_label, s.ai_predictions->0->>'label', ''), ':', 1), '')
    ) AS "CategoryPredicted",

    COALESCE(
        s.predicted_misconception,
        NULLIF(split_part(COALESCE(s.predicted_label, s.ai_predictions->0->>'label', ''), ':', 2), ''),
        s.top_misconception
    ) AS "MisconceptionPredicted",

    s.confidence AS "AIConfidence",

    -- ===========================================
    -- TEACHER RANKINGS (what the teacher selected)
    -- ===========================================
    NULLIF(split_part(COALESCE(tf.correct_prediction_label, ''), ':', 1), '') AS "CategoryRanked",

    COALESCE(
        NULLIF(split_part(COALESCE(tf.correct_prediction_label, ''), ':', 2), ''),
        tf.additional_prediction
    ) AS "MisconceptionRanked",

    tf.additional_prediction AS "TeacherAdditionalPrediction",

    -- ===========================================
    -- METADATA
    -- ===========================================
    s.question_source AS "QuestionSource",
    tf.id IS NOT NULL AS "TeacherReviewed",
    s.created_at AS "SubmittedAt",
    tf.created_at AS "ReviewedAt"

FROM public.submissions s
LEFT JOIN public.teacher_feedback tf ON tf.submission_id = s.id;

-- Grant access to the retraining_dataset view for anonymous/service role
-- This makes it accessible for data export without authentication
GRANT SELECT ON public.retraining_dataset TO anon;
GRANT SELECT ON public.retraining_dataset TO authenticated;
GRANT SELECT ON public.retraining_dataset TO service_role;
