# Visual audit and design system

| Screen family | Finding | Current cycle |
| --- | --- | --- |
| Home and onboarding | Repeated image cards gave every action equal weight while leaving ledger data hidden. Guest pages offered protected action links. | Signed-in overview shows date controls, three financial totals, a readable monthly comparison and recent activity. Guests get direct signup and sign-in actions. |
| Expense list | A populated account rendered all rows and offered no search. | 40-row pages, keyboard-accessible table scrolling, search, visible counts and empty results. |
| Income and goal lists | Basic Bootstrap tables provide limited context and progress information. | Income uses the ledger table. Goals use cards with accessible progress and an edit action. |
| Sign-in and registration | Purple backdrop, translucent card, and forced page height diverged from the dashboard, particularly when a notice was present. | Shared auth template with dashboard surfaces, consistent fields, browser autocomplete, and responsive spacing. |
| Profile | The earlier photo backdrop and forced viewport height diverged from the account screens. | Shared account layout, dashboard styling, and labeled upload and password hints. |
| Financial creation forms | Older pages retain inline styling and mixed hierarchy. | Income, expenses, goals and recurring reminders share the same entry form and field styles. Invalid entries retain submitted values. |
| Forecast | Raw model output displayed several decimal places. | The forecast uses a dashboard panel with themed colors, an accessible values table and reduced-motion support. |
| Anomalies, clusters and spending reference | Tables and lists show results, with limited explanation and inconsistent density. | Analytics now share the dashboard table and panel styles. Cluster details can be expanded and the anomaly methods are discoverable. Model validation remains open. |

`static/finance.css` defines color, border and surface tokens, responsive dashboard layout, controls with 44 px targets, visible focus, an accessible table container and reduced-motion preferences. The trend uses labeled values and CSS bars, so account data is readable without JavaScript. The overview shows only metrics derived from the current ledger. Display formatting does not migrate stored financial values.

## Browser verification

In the local preview, the populated overview and page-one expense list had no document-level horizontal overflow at 375, 768 and 1440 CSS px. The last-three-month selection showed three monthly rows. Each expense page rendered 40 records, and searching for the synthetic refund description returned five matches. Checks used the seeded account containing 1569 expenses.
