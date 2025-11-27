# Vyber – Test Plan (QA Lead Draft)

## Test Approach
Manual functional testing of API + UI to validate MVP behavior.

## Test Matrix (High Level)
- Moods: happy, sad, calm, excited, neutral
- Endpoints: /mood_input, /recommendations, /feedback
- UI: mood buttons, text input, results list, feedback buttons

## Test Cases
| ID | Area | Scenario | Steps | Expected Result |
|---|---|---|---|---|
| TC-001 | Mood Mapping | Button: happy | Click happy → fetch recommendations | 5 movies, genres align with happy mapping |
| TC-002 | Mood Mapping | Button: sad | Click sad → fetch recommendations | 5 movies, drama/biography present |
| TC-003 | Mood Detection | Text: "I feel awesome" | POST /mood_input | mood = happy |
| TC-004 | Mood Detection | Text: "feeling down today" | POST /mood_input | mood = sad |
| TC-005 | Fallback | Text: gibberish "asd###" | POST /mood_input | mood = neutral |
| TC-006 | Recommendations | GET /recommendations?mood=calm | Call endpoint | 5 movies, genres from calm mapping |
| TC-007 | Explanations | Any mood | Call recommendations | Each item has short explanation referencing mood/genre |
| TC-008 | Feedback | 👍 | Click thumbs-up on a result | 200 OK from /feedback and row stored |
| TC-009 | Error Handling | Unknown mood | GET /recommendations?mood=xyz | 200 OK with neutral fallback |
| TC-010 | Performance | Any mood | Measure response | First result < 2 seconds on laptop |

## Bug Reporting
Open GitHub Issue with:
- Title: `[QA] <short description>`
- Steps to Reproduce
- Expected vs Actual
- Screenshot / logs
- Priority: P0 (blocker) / P1 (major) / P2 (minor)

## Test Data
See `/data/mood_genre_mapping.csv`. Use MovieLens/TMDb demo subset.

## Sign-off
QA Lead will sign off when all P0 and P1 issues are closed and acceptance criteria met.
