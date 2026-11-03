import { useEffect, useState } from "react";
import { desk, type JobDetail } from "../api";

/** Follow one job until it ends; null while there is none. */
export function useJob(id: string | null): JobDetail | null {
  const [job, setJob] = useState<JobDetail | null>(null);
  useEffect(() => {
    if (!id) {
      setJob(null);
      return undefined;
    }
    let stop = false;
    let timer = 0;
    const tick = async () => {
      try {
        const [found] = await desk<JobDetail[]>(`/api/jobs?id=${encodeURIComponent(id)}`);
        if (stop) return;
        setJob(found ?? null);
        if (found && found.state !== "running") return;
      } catch {
        /* keep polling */
      }
      if (!stop) timer = window.setTimeout(tick, 700);
    };
    void tick();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [id]);
  return job;
}

/** The newest job of a command, so a page re-opened mid-run picks it up. */
export async function latestJob(command: string): Promise<JobDetail | null> {
  try {
    const jobs = await desk<JobDetail[]>("/api/jobs");
    const mine = jobs.filter((j) => j.command === command);
    return mine.length ? mine[mine.length - 1] : null;
  } catch {
    return null;
  }
}
