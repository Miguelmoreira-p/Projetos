-- =============================================================================
-- EduPhone Guard — Schema PostgreSQL/Supabase
-- =============================================================================
-- Este script cria as tabelas, índices e políticas de Row Level Security
-- (RLS) do MVP. Execute no SQL Editor do Supabase (ou via `supabase db push`
-- com as migrations equivalentes).
--
-- IMPORTANTE: RLS complementa (não substitui) a autorização feita no
-- backend (app/security/rbac.py). As duas camadas devem concordar.
-- =============================================================================

create extension if not exists "pgcrypto";

-- -----------------------------------------------------------------------------
-- schools
-- -----------------------------------------------------------------------------
create table if not exists schools (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    created_at timestamptz not null default now()
);

-- -----------------------------------------------------------------------------
-- users (perfil complementar ao auth.users do Supabase)
-- -----------------------------------------------------------------------------
-- O papel (role) e a escola do usuário também são espelhados em
-- auth.users.raw_app_meta_data (via Custom Access Token Hook) para que
-- estejam disponíveis como claims no JWT, mas mantemos esta tabela como
-- fonte de verdade administrável pela UI.
create table if not exists user_profiles (
    id uuid primary key references auth.users (id) on delete cascade,
    school_id uuid references schools (id) on delete set null,
    role text not null check (role in ('ADMIN', 'DIRECTOR', 'REVIEWER')),
    full_name text,
    created_at timestamptz not null default now()
);

-- -----------------------------------------------------------------------------
-- cameras
-- -----------------------------------------------------------------------------
create table if not exists cameras (
    id uuid primary key default gen_random_uuid(),
    school_id uuid not null references schools (id) on delete cascade,
    name text not null,
    location text not null,
    stream_url text, -- nunca exposto diretamente via API pública
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create index if not exists idx_cameras_school on cameras (school_id);

-- -----------------------------------------------------------------------------
-- events
-- -----------------------------------------------------------------------------
create table if not exists events (
    id uuid primary key default gen_random_uuid(),
    school_id uuid not null references schools (id) on delete cascade,
    camera_id uuid not null references cameras (id) on delete cascade,
    created_at timestamptz not null default now(),
    detected_at timestamptz not null,
    confidence numeric(4, 3) not null check (confidence >= 0 and confidence <= 1),
    status text not null default 'PENDING'
        check (status in ('PENDING', 'CONFIRMED', 'FALSE_POSITIVE', 'ARCHIVED')),
    video_path text, -- caminho dentro do bucket privado, nunca URL pública
    retention_until timestamptz,
    detection_metadata jsonb not null default '{}'::jsonb
);

create index if not exists idx_events_school on events (school_id);
create index if not exists idx_events_status on events (status);
create index if not exists idx_events_retention on events (retention_until);

-- -----------------------------------------------------------------------------
-- event_reviews
-- -----------------------------------------------------------------------------
create table if not exists event_reviews (
    id uuid primary key default gen_random_uuid(),
    event_id uuid not null references events (id) on delete cascade,
    reviewer_id uuid not null references auth.users (id),
    status text not null check (status in ('PENDING', 'CONFIRMED', 'FALSE_POSITIVE', 'ARCHIVED')),
    notes text,
    created_at timestamptz not null default now()
);

create index if not exists idx_event_reviews_event on event_reviews (event_id);

-- -----------------------------------------------------------------------------
-- audit_logs
-- -----------------------------------------------------------------------------
create table if not exists audit_logs (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users (id),
    action text not null,
    resource_type text not null,
    resource_id text,
    created_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb
);

create index if not exists idx_audit_logs_created_at on audit_logs (created_at desc);

-- -----------------------------------------------------------------------------
-- system_settings (linha única "default", editável pelo ADMIN)
-- -----------------------------------------------------------------------------
create table if not exists system_settings (
    id text primary key default 'default',
    min_confidence numeric(4, 3),
    min_detection_frames int,
    window_seconds numeric(6, 2),
    cooldown_seconds numeric(6, 2),
    retention_days int,
    updated_at timestamptz not null default now()
);

-- =============================================================================
-- Row Level Security
-- =============================================================================
alter table schools enable row level security;
alter table user_profiles enable row level security;
alter table cameras enable row level security;
alter table events enable row level security;
alter table event_reviews enable row level security;
alter table audit_logs enable row level security;
alter table system_settings enable row level security;

-- Função utilitária: papel do usuário autenticado atual, a partir do JWT.
create or replace function current_user_role() returns text as $$
    select coalesce(auth.jwt() -> 'app_metadata' ->> 'role', '');
$$ language sql stable;

create or replace function current_user_school_id() returns uuid as $$
    select nullif(auth.jwt() -> 'app_metadata' ->> 'school_id', '')::uuid;
$$ language sql stable;

-- --- cameras: usuários só veem câmeras da própria escola; ADMIN vê tudo.
create policy cameras_select on cameras for select
    using (current_user_role() = 'ADMIN' or school_id = current_user_school_id());

create policy cameras_write_admin on cameras for insert
    with check (current_user_role() = 'ADMIN');

create policy cameras_update_admin on cameras for update
    using (current_user_role() = 'ADMIN');

-- --- events: mesma regra de escopo por escola.
create policy events_select on events for select
    using (current_user_role() = 'ADMIN' or school_id = current_user_school_id());

-- Inserção de eventos é feita exclusivamente pelo backend com a service
-- role key (que ignora RLS). Não criamos policy de insert para clientes
-- autenticados comuns.

create policy events_update_reviewers on events for update
    using (
        current_user_role() in ('ADMIN', 'DIRECTOR', 'REVIEWER')
        and (current_user_role() = 'ADMIN' or school_id = current_user_school_id())
    );

-- --- event_reviews: visível para quem pode ver o evento correspondente.
create policy event_reviews_select on event_reviews for select
    using (
        exists (
            select 1 from events e
            where e.id = event_reviews.event_id
              and (current_user_role() = 'ADMIN' or e.school_id = current_user_school_id())
        )
    );

create policy event_reviews_insert on event_reviews for insert
    with check (current_user_role() in ('ADMIN', 'DIRECTOR', 'REVIEWER'));

-- --- audit_logs: apenas ADMIN pode ler.
create policy audit_logs_select_admin on audit_logs for select
    using (current_user_role() = 'ADMIN');

-- --- system_settings: leitura para ADMIN/DIRECTOR, escrita só para ADMIN.
create policy system_settings_select on system_settings for select
    using (current_user_role() in ('ADMIN', 'DIRECTOR'));

create policy system_settings_write_admin on system_settings for all
    using (current_user_role() = 'ADMIN')
    with check (current_user_role() = 'ADMIN');

-- --- user_profiles: cada usuário vê seu próprio perfil; ADMIN vê todos.
create policy user_profiles_select on user_profiles for select
    using (id = auth.uid() or current_user_role() = 'ADMIN');
