import { useMemo, useState } from "react";

type RemotePreference = "no_preference" | "preferred" | "required";

type UserProfile = {
  skills: string[];
  preferred_job_types: string[];
  preferred_locations: string[];
  salary_range: {
    min?: number;
    max?: number;
  };
  remote_preference: RemotePreference;
};

type RecommendRequest = {
  user_profile: UserProfile;
};

type JobRecommendation = {
  job_id?: string;
  rank?: number;
  job_title?: string;
  job_title_short?: string;
  company_name?: string;
  job_location?: string;
  normalized_salary?: number;
  job_work_from_home?: boolean;
  job_health_insurance?: boolean;
  days_since_posted?: number;
  similarity_score?: number;
  ranking_score?: number;
  all_skills?: string[];
  algorithm?: string;
};

function parseCommaOrNewlineList(value: string): string[] {
  return value
    .split(/,|\n/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function formatSalary(value: unknown): string {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n) || n <= 0) return "-";
  return n.toFixed(0);
}

const App = () => {
  const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

  const [skills, setSkills] = useState("");
  const [preferredJobTypes, setPreferredJobTypes] = useState("");
  const [preferredLocations, setPreferredLocations] = useState("");
  const [salaryMin, setSalaryMin] = useState<string>("");
  const [salaryMax, setSalaryMax] = useState<string>("");
  const [remotePreference, setRemotePreference] =
    useState<RemotePreference>("no_preference");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<
    JobRecommendation[] | null
  >(null);

  const requestBody: RecommendRequest = useMemo(() => {
    const min = salaryMin.trim() === "" ? undefined : Number(salaryMin);
    const max = salaryMax.trim() === "" ? undefined : Number(salaryMax);

    return {
      user_profile: {
        skills: parseCommaOrNewlineList(skills),
        preferred_job_types: parseCommaOrNewlineList(preferredJobTypes),
        preferred_locations: parseCommaOrNewlineList(preferredLocations),
        salary_range: {
          ...(min !== undefined && Number.isFinite(min) ? { min } : {}),
          ...(max !== undefined && Number.isFinite(max) ? { max } : {}),
        },
        remote_preference: remotePreference,
      },
    };
  }, [
    skills,
    preferredJobTypes,
    preferredLocations,
    salaryMin,
    salaryMax,
    remotePreference,
  ]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();

    setLoading(true);
    setError(null);
    setRecommendations(null);

    try {
      const res = await fetch(`${API_BASE_URL}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(text || `Request failed with status ${res.status}`);
      }

      const data = (await res.json()) as JobRecommendation[];
      setRecommendations(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to fetch recommendations",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl p-6">
      <h1 className="mb-6 text-2xl font-semibold">Job Recommendations</h1>

      <form onSubmit={onSubmit} className="space-y-6 rounded-lg border p-4">
        <div className="space-y-2">
          <label className="block text-sm font-medium">
            Skills (comma or newline separated)
          </label>
          <textarea
            className="w-full rounded border p-2"
            rows={3}
            value={skills}
            onChange={(e) => setSkills(e.target.value)}
            placeholder="e.g. React, TypeScript, SQL"
            required={parseCommaOrNewlineList(skills).length > 0}
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium">
            Preferred job types
          </label>
          <textarea
            className="w-full rounded border p-2"
            rows={2}
            value={preferredJobTypes}
            onChange={(e) => setPreferredJobTypes(e.target.value)}
            placeholder="e.g. backend, frontend, data engineer"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium">
            Preferred locations
          </label>
          <textarea
            className="w-full rounded border p-2"
            rows={2}
            value={preferredLocations}
            onChange={(e) => setPreferredLocations(e.target.value)}
            placeholder="e.g. San Francisco, Remote, New York"
          />
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="block text-sm font-medium">Salary min</label>
            <input
              className="w-full rounded border p-2"
              inputMode="numeric"
              value={salaryMin}
              onChange={(e) => setSalaryMin(e.target.value)}
              placeholder="e.g. 80000"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium">Salary max</label>
            <input
              className="w-full rounded border p-2"
              inputMode="numeric"
              value={salaryMax}
              onChange={(e) => setSalaryMax(e.target.value)}
              placeholder="e.g. 160000"
            />
          </div>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium">Remote preference</label>
          <select
            className="w-full rounded border p-2"
            value={remotePreference}
            onChange={(e) =>
              setRemotePreference(e.target.value as RemotePreference)
            }
          >
            <option value="no_preference">No preference</option>
            <option value="preferred">Preferred</option>
            <option value="required">Required</option>
          </select>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="submit"
            className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-60"
            disabled={loading}
          >
            Get Recommendations
          </button>
        </div>
      </form>

      {error ? (
        <div className="mt-5 rounded border border-red-300 bg-red-50 p-3 text-red-800">
          {error}
        </div>
      ) : null}

      {recommendations ? (
        <div className="mt-6">
          {recommendations.length === 0 ? (
            <div className="rounded border p-4 text-sm text-gray-700">
              No recommendations found.
            </div>
          ) : (
            <div className="overflow-x-auto rounded-lg border">
              <table className="min-w-full border-collapse text-left text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="border-b px-3 py-2">Rank</th>
                    <th className="border-b px-3 py-2">Job Title</th>
                    <th className="border-b px-3 py-2">Company</th>
                    <th className="border-b px-3 py-2">Location</th>
                    <th className="border-b px-3 py-2">Remote</th>
                    <th className="border-b px-3 py-2">Salary</th>
                    <th className="border-b px-3 py-2">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {recommendations.map((job) => {
                    const title = job.job_title_short || job.job_title || "N/A";
                    const score =
                      typeof job.similarity_score === "number"
                        ? job.similarity_score
                        : typeof job.ranking_score === "number"
                          ? job.ranking_score
                          : undefined;

                    return (
                      <tr
                        key={job.job_id ?? `${job.rank ?? 0}-${title}`}
                        className="hover:bg-gray-50"
                      >
                        <td className="border-b px-3 py-2">
                          {job.rank ?? "-"}
                        </td>
                        <td className="border-b px-3 py-2 font-medium">
                          {title}
                        </td>
                        <td className="border-b px-3 py-2">
                          {job.company_name ?? "-"}
                        </td>
                        <td className="border-b px-3 py-2">
                          {job.job_location ?? "-"}
                        </td>
                        <td className="border-b px-3 py-2">
                          {job.job_work_from_home ? "Yes" : "No"}
                        </td>
                        <td className="border-b px-3 py-2">
                          {formatSalary(job.normalized_salary)}
                        </td>
                        <td className="border-b px-3 py-2">
                          {typeof score === "number" ? score.toFixed(3) : "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : null}

      {loading && !recommendations ? (
        <div className="mt-6 text-sm text-gray-600">Loading...</div>
      ) : null}
    </div>
  );
};

export default App;
