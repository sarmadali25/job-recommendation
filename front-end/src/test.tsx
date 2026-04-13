import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from './App'

describe('frontend-backend integration flow', () => {
  it('submits user profile, calls backend /recommendations, and renders returned jobs', async () => {
    const mockRecommendations = [
      {
        job_id: 'job-1',
        rank: 1,
        job_title: 'Backend Engineer',
        company_name: 'Acme',
        job_location: 'Remote',
        job_work_from_home: true,
        normalized_salary: 120000,
        similarity_score: 0.91,
        algorithm: 'content',
      },
    ]

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockRecommendations,
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    fireEvent.change(
      screen.getByPlaceholderText(
        /e\.g\. SQL, Python, Tableau, machine learning/i,
      ),
      { target: { value: 'Python, SQL' } },
    )

    fireEvent.click(screen.getByRole('button', { name: /Get Recommendations/i }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toContain('/recommendations')
    expect(options.method).toBe('POST')
    expect(options.headers).toEqual({ 'Content-Type': 'application/json' })
    const body = String(options.body)
    expect(body).toContain('"user_profile"')
    expect(body).toContain('"skills":["Python","SQL"]')
    expect(body).toContain('"remote_preference":"no_preference"')

    expect(await screen.findByText('Backend Engineer')).toBeInTheDocument()
    expect(screen.getByText('Acme')).toBeInTheDocument()
    expect(screen.getByText('0.910')).toBeInTheDocument()
  })
})
