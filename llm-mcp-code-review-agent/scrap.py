
@app.route("/webhook", methods=["POST"])
async def webhook(request: Request, x_github_event: str = Header(...)):
    try:
        x_github_event = request.headers.get("X-GitHub-Event")
        log.info(f"Received webhook event: {x_github_event}")
        data = await request.json()
    except Exception as e:
        log.error(f"Error parsing JSON: {e}")
        return JSONResponse(content={"status": "error", "message": "Invalid JSON"}, status_code=400)
    log.info(f"Webhook data: {data}")
        # Handle PR review comment events
    if x_github_event == "pull_request_review_comment":
        comment_body = data.get("comment", {}).get("body", "")
        if "@code_review" in comment_body:
            repo_full_name = data["repository"]["full_name"]               # e.g. alexcpn/accelerate-test
            pr_url = data["comment"]["pull_request_url"]                   # e.g. .../pulls/1
            pr_number = int(pr_url.split("/")[-1])
            repo_url = f"https://github.com/{repo_full_name}"

            log.info(f"Triggered code review on {repo_url} PR #{pr_number}")

            review_comment = await main(repo_url, pr_number) or "No issues found."

            # Post back to the same thread
            comment_url = data["comment"]["url"]
            headers = {
                "Authorization": f"token {GITLAB_TOKEN}",
                "Accept": "application/vnd.github+json"
            }
            post_response = requests.post(
                comment_url,
                headers=headers,
                json={"body": f"AI Code Review:\n```\n{review_comment}\n```"}
            )
            log.info(f"Posted review result: {post_response.status_code}")
            return JSONResponse(content={"status": "review triggered"})
        
    return JSONResponse(content={"status": "ok"})