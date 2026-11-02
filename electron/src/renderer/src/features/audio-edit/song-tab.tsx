import { EditWorkspace } from './edit-workspace';

export function SongTab(props: { sessionId: string | null; onSessionChange: (id: string | null) => void }) {
  return <EditWorkspace mode="song" {...props} />;
}
