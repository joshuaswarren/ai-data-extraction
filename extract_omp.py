#!/usr/bin/env python3
"""
Extract ALL omp (Oh My Pi) chat/agent data from all projects.

Storage layout:
  ~/.omp/agent/sessions/<slugified-cwd>/<ISO8601>_<uuid>.jsonl   session transcript
  ~/.omp/agent/sessions/<slugified-cwd>/<ISO8601>_<uuid>/        sidecar tool logs
  ~/.omp/agent/history.db                                        prompt history (FTS)

Transcript records are newline-delimited JSON with a top-level "type":
  session / title / title_change                     -> session metadata
  model_change / mode_change / thinking_level_change -> runtime state
  message                                            -> {"message": {"role", "content"}}
        roles: user | assistant | developer | toolResult
        content part types: text | thinking | toolCall
  custom / custom_message                            -> extension payloads
  compaction / branch_summary                        -> context compaction markers

Forked sessions keep both descendants in one append-only file, linked by
id/parentId. Each unique root-to-leaf message path is emitted as its own
conversation.

Includes: user/assistant messages, thinking blocks, tool calls + arguments,
tool results, model attribution, compaction boundaries, project path.
No dependencies beyond the Python 3 standard library.
"""

import json
import os
import platform
import sqlite3
from datetime import datetime
from pathlib import Path

# Runtime-state records: no conversational payload.
STATE_TYPES = {'model_change', 'mode_change', 'thinking_level_change'}


def find_omp_installations():
    """Find all omp data directories."""
    system = platform.system()
    home = Path.home()
    candidates = []

    # Explicit override honoured by the omp harness.
    env_dir = os.environ.get('OMP_HOME') or os.environ.get('OMP_DATA_DIR')
    if env_dir:
        candidates.append(Path(env_dir))

    # Primary location on every platform.
    candidates.append(home / '.omp')

    if system == 'Darwin':
        candidates += [
            home / 'Library/Application Support/omp',
            home / '.config/omp',
        ]
    elif system == 'Linux':
        candidates += [
            home / '.local/share/omp',
            home / '.config/omp',
        ]
    elif system == 'Windows':
        candidates += [
            Path(os.environ.get('APPDATA', home / 'AppData/Roaming')) / 'omp',
            Path(os.environ.get('LOCALAPPDATA', home / 'AppData/Local')) / 'omp',
        ]

    found = []
    for c in candidates:
        # Accept either <dir>/agent/sessions or <dir>/sessions.
        for root in (c / 'agent', c):
            if (root / 'sessions').is_dir() and root not in found:
                found.append(root)
    return found


def _parts(content):
    """Normalise a message content field into a list of part dicts."""
    if isinstance(content, list):
        return [p for p in content if isinstance(p, dict)]
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]
    return []


def _text(parts):
    return '\n'.join(p.get('text', '') for p in parts if p.get('type') == 'text').strip()


def _load_records(jsonl_file):
    records = []
    parse_errors = 0
    with open(jsonl_file, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
    return records, parse_errors


def _file_meta(records):
    session_id = title = project_path = created_at = None
    for obj in records:
        rtype = obj.get('type')
        if rtype == 'session':
            session_id = obj.get('id') or session_id
            project_path = obj.get('cwd') or project_path
            created_at = obj.get('timestamp') or created_at
            title = obj.get('title') or title
        elif rtype in ('title', 'title_change'):
            title = obj.get('title') or title
    return session_id, title, project_path, created_at


def _ancestor_chain(leaf_id, by_id):
    chain = []
    cur = leaf_id
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        rec = by_id.get(cur)
        if not rec:
            break
        chain.append(rec)
        cur = rec.get('parentId')
    chain.reverse()
    return chain


def _conversation_from_records(
    records,
    jsonl_file,
    parse_errors,
    session_id,
    title,
    project_path,
    created_at,
    include_thinking=True,
    include_developer=True,
):
    """Build one conversation from an ordered record list."""
    messages = []
    models = []
    current_model = None
    compactions = 0

    for obj in records:
        rtype = obj.get('type')

        if rtype == 'session' or rtype in ('title', 'title_change'):
            continue

        if rtype in STATE_TYPES:
            if rtype == 'model_change':
                # Older builds emit "model"; newer ones "modelId" (+ "provider").
                current_model = obj.get('model') or obj.get('modelId')
                provider = obj.get('provider')
                if current_model and provider and '/' not in current_model:
                    current_model = f'{provider}/{current_model}'
                if current_model and current_model not in models:
                    models.append(current_model)
            continue

        if rtype in ('compaction', 'branch_summary'):
            compactions += 1
            messages.append({
                'role': 'system',
                'event': rtype,
                'content': obj.get('summary') or obj.get('text') or '',
                'timestamp': obj.get('timestamp'),
            })
            continue

        if rtype == 'custom_message':
            # Extension-authored turns keep content at the top level, not
            # inside a nested "message" object.
            content = obj.get('content')
            text = content if isinstance(content, str) else _text(_parts(content))
            if not text:
                continue
            messages.append({
                'role': obj.get('role') or obj.get('attribution') or 'system',
                'content': text,
                'timestamp': obj.get('timestamp'),
                'id': obj.get('id'),
                'parent_id': obj.get('parentId'),
                'custom': True,
                'custom_type': obj.get('customType'),
                'display': obj.get('display'),
                'details': obj.get('details'),
            })
            continue

        if rtype != 'message':
            # 'custom' and anything else is extension-private; skip.
            continue

        inner = obj.get('message') or {}
        role = inner.get('role')
        parts = _parts(inner.get('content'))
        if not role or not parts:
            continue

        if role == 'developer' and not include_developer:
            continue

        if role == 'toolResult':
            result = {
                'tool_call_id': inner.get('toolCallId') or obj.get('toolCallId'),
                'content': _text(parts),
                'timestamp': obj.get('timestamp'),
            }
            # Attach to the owning assistant turn when possible.
            for prev in reversed(messages):
                if prev.get('role') == 'assistant':
                    prev.setdefault('tool_results', []).append(result)
                    break
            else:
                messages.append({'role': 'toolResult', **result})
            continue

        msg = {
            'role': role,
            'content': _text(parts),
            'timestamp': obj.get('timestamp'),
            'id': obj.get('id'),
            'parent_id': obj.get('parentId'),
        }

        if role == 'assistant':
            msg['model'] = inner.get('model') or current_model
            thinking = [p.get('thinking', '') for p in parts if p.get('type') == 'thinking']
            if thinking and include_thinking:
                msg['thinking'] = '\n'.join(t for t in thinking if t)
            tool_calls = [
                {
                    'id': p.get('id') or p.get('toolCallId'),
                    'name': p.get('name') or p.get('tool'),
                    'arguments': p.get('arguments', p.get('args')),
                }
                for p in parts if p.get('type') == 'toolCall'
            ]
            if tool_calls:
                msg['tool_calls'] = tool_calls

        if msg['content'] or msg.get('tool_calls') or msg.get('thinking'):
            messages.append(msg)

    if not messages:
        return None

    sidecar = jsonl_file.with_suffix('')
    tool_logs = sorted(p.name for p in sidecar.iterdir()) if sidecar.is_dir() else []

    if session_id is None:
        session_id = jsonl_file.stem.split('_', 1)[-1]

    return {
        'messages': messages,
        'source': 'omp',
        'session_id': session_id,
        'name': title,
        'project_path': project_path,
        'project_slug': jsonl_file.parent.name,
        'created_at': created_at,
        'models': models,
        'compactions': compactions,
        'tool_logs': tool_logs,
        'source_file': str(jsonl_file),
        'parse_errors': parse_errors,
    }


def extract_session(jsonl_file, include_thinking=True, include_developer=True):
    """Extract one omp transcript into one or more normalised conversations.

    Returns a list: a linear session yields one item; a forked session yields
    one item per unique message path.
    """
    records, parse_errors = _load_records(jsonl_file)
    session_id, title, project_path, created_at = _file_meta(records)

    by_id = {r['id']: r for r in records if r.get('id')}
    referenced = {r.get('parentId') for r in records if r.get('parentId')}
    leaves = [r['id'] for r in records if r.get('id') and r['id'] not in referenced]
    # Unique, file order. A record can only appear once in by_id.
    seen = set()
    ordered_leaves = []
    for leaf in leaves:
        if leaf not in seen:
            seen.add(leaf)
            ordered_leaves.append(leaf)

    chains = [records] if not ordered_leaves else [
        _ancestor_chain(leaf, by_id) for leaf in ordered_leaves
    ]
    leaf_for_chain = [None] if not ordered_leaves else ordered_leaves

    convos = []
    seen_sig = set()
    kept_leaves = []
    for chain, leaf in zip(chains, leaf_for_chain):
        conv = _conversation_from_records(
            chain,
            jsonl_file,
            parse_errors,
            session_id,
            title,
            project_path,
            created_at,
            include_thinking=include_thinking,
            include_developer=include_developer,
        )
        if not conv:
            continue
        sig = tuple(m.get('id') for m in conv['messages'])
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        convos.append(conv)
        kept_leaves.append(leaf)

    if len(convos) > 1:
        base = session_id or jsonl_file.stem.split('_', 1)[-1]
        for conv, leaf in zip(convos, kept_leaves):
            conv['session_id'] = f'{base}:{leaf}' if leaf else conv['session_id']

    return convos


def extract_prompt_history(root):
    """Extract the standalone prompt history database (best effort).

    omp runs history.db in WAL mode. sqlite3.Connection.backup() copies under a
    read transaction, so the snapshot cannot mix an old main file with a rotated
    WAL. Live files are never written.
    """
    db = root / 'history.db'
    if not db.exists():
        return []
    rows = []
    src = snap = None
    try:
        src = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        snap = sqlite3.connect(':memory:')
        src.backup(snap)
        src.close()
        src = None
        snap.row_factory = sqlite3.Row
        for r in snap.execute('SELECT * FROM history'):
            rows.append(dict(r))
    except (sqlite3.Error, OSError) as e:
        print(f'   ⚠️  history.db unreadable: {e}')
    finally:
        if src is not None:
            src.close()
        if snap is not None:
            snap.close()
    return rows


def main():
    print('=' * 80)
    print('OMP (OH MY PI) COMPLETE DATA EXTRACTION')
    print('=' * 80)
    print()

    print('🔍 Searching for omp installations...')
    installations = find_omp_installations()
    if not installations:
        print('❌ No omp installations found! (expected ~/.omp/agent/sessions)')
        return

    print(f'✅ Found {len(installations)} installation(s):')
    for inst in installations:
        print(f'   - {inst}')
    print()

    all_conversations = []
    project_stats = {}
    history_rows = []

    for root in installations:
        print(f'📂 Processing: {root}')
        transcripts = sorted((root / 'sessions').glob('*/*.jsonl'))
        # Skip extension sidecar transcripts (e.g. __advisor.jsonl).
        transcripts = [t for t in transcripts if not t.name.startswith('__')]
        print(f'   {len(transcripts)} transcript file(s)')

        for t in transcripts:
            try:
                convs = extract_session(t)
            except OSError as e:
                print(f'   Error reading {t}: {e}')
                continue
            for conv in convs:
                conv['installation'] = str(root)
                all_conversations.append(conv)
                key = conv.get('project_path') or conv['project_slug']
                project_stats[key] = project_stats.get(key, 0) + 1

        history_rows.extend(extract_prompt_history(root))

    print()
    print('=' * 80)
    print('EXTRACTION COMPLETE')
    print('=' * 80)
    print(f'Total conversations: {len(all_conversations):,}')
    print(f'Prompt history rows: {len(history_rows):,}')

    if not all_conversations and not history_rows:
        print('No conversations or prompt history found!')
        return

    output_dir = Path('extracted_data')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if all_conversations:
        total_messages = sum(len(c['messages']) for c in all_conversations)
        with_tools = sum(
            1 for c in all_conversations
            if any('tool_calls' in m or 'tool_results' in m for m in c['messages'])
        )
        with_thinking = sum(
            1 for c in all_conversations if any('thinking' in m for m in c['messages'])
        )
        complete = sum(
            1 for c in all_conversations
            if any(m['role'] == 'assistant' for m in c['messages'])
        )

        print(f'Complete conversations: {complete:,}')
        print(f'Total messages: {total_messages:,}')
        print(f'With tool calls/results: {with_tools:,}')
        print(f'With thinking blocks: {with_thinking:,}')
        print()

        print('Breakdown by project (top 20):')
        for proj, count in sorted(project_stats.items(), key=lambda x: -x[1])[:20]:
            print(f'  {str(proj)[:50]:50} {count:5,} conversations')
        print()

        output_file = output_dir / f'omp_conversations_{timestamp}.jsonl'
        with open(output_file, 'w') as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')

        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f'✅ Saved to: {output_file}')
        print(f'   Size: {size_mb:.2f} MB')
        print('   Format: JSONL (one conversation per line)')
    else:
        print('No conversations found!')
        print()

    if history_rows:
        # Not *.jsonl: extract_all.sh globs extracted_data/*.jsonl, and
        # filter_privacy.py / corpus_to_skills.py rglob("*.jsonl") recursively.
        hist_file = output_dir / f'omp_prompt_history_{timestamp}.json'
        with open(hist_file, 'w') as f:
            json.dump(history_rows, f, ensure_ascii=False, default=str)
        print(f'✅ Saved to: {hist_file}')
        print('   (JSON array, not jsonl: excluded from conversation corpus globs)')


if __name__ == '__main__':
    main()
