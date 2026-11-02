import { EditWorkspace } from './edit-workspace';

export function SpeechTab(props: { sessionId: string | null; onSessionChange: (id: string | null) => void }) {
  return <EditWorkspace mode="speech" {...props} />;
}
