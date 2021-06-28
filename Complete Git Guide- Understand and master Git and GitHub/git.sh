## 1. Introduction to the Git and GitHub
## 2. Installation of the Git and configuration of the Shell
## 3. Basic Shell commands
    mkdir git_proj
    cd git_proj
## 4. How Git works under the hood
    git init
    # low level hash fucntions for git fs
        git hash-object -w 
            # <type> <length>'\0'<content>; "blob 11\0Hello, Git"
                # compressed binary format
        git cat-file -p -s -t <hash>
        git mktree
            # permissions are git fs perms
            # types blob, tree, 
            # <permissions> <type> <hash>\t<name>
            # <permissions> <type> <hash>\t<name>
    # staging area, index
        git ls-files -s # git's ls; ?staging area
        git read-tree <hash> # git -> staging
        git checkout-index -a # staging -> working
        git status
## 5. Basic Git operations
    # commit; just points to tree, like wrapper
        # tree <hash>
        # parent <parent>
        # author <name> <email> <time from epoch> <timezone>
        # comitter <name> <email> <time from epoch> <timezone>
        # 
        # <desc>
        git config --global user.name <name>
        git config --global user.mail <mail>
        git config --list

        git commit -m <desc>

        git status
        git add . # . to stage all, else use names
            # untracked (new file), modified, staged, unmodified
            git rm --cached <file>
        git commit
        git log
        git checkout
        # same file hash might be used in different commit trees, hence git doesnt create new info for same object across commits
## 6. Git branches and HEAD
    git checkout # go to specific version
    # .git/refs/heads - pointers to branches
    # HEAD points to current branch
    git checkout <branch/sha1> 
        # replace staging, working dir
        # only changes HEAD, not branch
        # will also change git log op and stuff, literal travel of time    
    # detached HEAD - pointing to commit, not branch
        # commits on detached HEAD might be deleted by git (?)
    git branch # list
    git branch <name> # -d for delete, only merged branches, -D for unmerged
    git checkout -b <branch> # create and checkout
    # git reuses blobs with same contents whenever possible
## 7. Cloning, exploring and modifying public repositories
    git clone <.git url>
    # git might pack /objects to optimize space
    # origin/master - origin denotes its remote
    git diff
        #  @@ -a,b +c,d @@ <start text> -> old-a line#,b #of lines, c,d for new
    git commit -a # add and commit
## 8. Merging branches
    # fast forward merge
        # only when receiving branch has no extra commits after branch-point
        # just brings recv-branch pointer to target
        git checkout <recv-branch>
        git merge <feat-branch>
    # 3 way merge
        # creates a new merge commit (two parents) and moves master
            # two or more parents also possible
        # feat-branch pointer doesnt change, only master changes
        # 3 way
            # <<<<< HEAD
            # <head content>
            # =======
            # <feat br content>
            # >>>>>> <feat br>
            # change above full section to whats finally to be committed
        git ls-files -s
            # 1 common ancestry
            # 2 recv br
            # 3 feat br
## 9. GitHub and remote repositories
    # https://www.github.com
## 10. Git push, fetch and pull
    # default remote repo - origin
    git remote -v # deafult clones/lists only local branches
    git branch
        # -a all
        # -r only remote
        # -vv to check tracking branches
    # To track only-remote branch, simply checkout to it
        # if deleted, checkout again to track again
    git remote show origin
    git fetch # fetch from remote, non-destructive
    git remote prune origin # sync branch deletions with origin
    git pull -vv
        # fetch + merge FETCH_HEAD
    git push -v
    git push --set-upstream origin <new-branch>    
    git remote update origin --prune # update about deleted remote branches
    git push origin -d <branch> # remove remote branch
    git show-ref [<branch>]
## 11. Pull requests
    # not git's feature, but github's
    git commit --amend # Overwrite commit, --author="" -> change author
## 12. Forks and contribution to the public repositories
    git remote add upstream <https://...git>
        # upstream name is usually for forked-from (parent) repo
    git fetch <remote(upstream)> <branch>
    git pull upstream master
    # git remote add, pull, push -> to sync origin with upstream
    # pull request for syncing upstream with origin (not gits feature, but githubs)
## 13. Git Tags
    git tag # tag ls
    git tag <tag> # git tag -v <tag>
    git show <tag>
    git lg # simple git log
    git tag -a <tag> -m <message>
    git push --tags
    git push <tag>
## 14. Rebasing
    # Destructive (kinda)
    # Instead of 3 way merge
    # it moves <feature> root to current <base> commit
    # duplicates commits on top of it
    # and fast forward merges
    git checkout <feature>
    git rebase <base branch>
    git checkout <base branch>
    git merge <feature>
## 15. Ignoring files in Git
    # .gitignore
    # to not consider files
    # should be committed
    # <file>
    # <folder>/
    # *.<extension>
    # <regex>
    git rm --cached <file> # removes from staging area
    # gitignore.io
    # github.com/github/gitignore
## 16. Detached HEAD
    git checkout <commit-hash>
    git commit # checkout to another branch to forgo these changes (will be garbage collected later)
    git branch <branch> # else might be garbage collected
## 17. Advanced Git
    git 
        # lg
        # log
            # --oneline
            # --graph
            # --stat
            # -p
            # -<num>
            # --grep="<string>"
            # --pretty=format:"<format>" 
            # --merges
            # --no-merges
        # shortlog
            # -n
            # -s
            # -e
            # --author=<author> (<author> is regex by default)
        # reset ;changes branch to commit too
            # <hash> ;uncommits, unstages, unchanged working dir
                # --soft ;uncommits, stages, unchanged working dir
                # --hard ;uncommits, unstage, changed working dir
                    # --hard, followed by --hard resets the reset (commits are snapshots)
            # HEAD~<num>
        # revert ;uncommits and commits the uncommitting
            # <hash> ;revert specific commit
            # HEAD ;last commit
        # commit
            # --amend ;recommit, discard the previous wrong commit
        # cherry-pick ;apply changes of a commit to current working dir
            # <hash> ;cherry-picks and commits
                # --no-commit
        # reflog ;local git history ;stores for 90 days
            # show <branch>
            # git checkout HEAD@{6} ;reflog to 6th in HEAD
        # stash ;stashes
            # pop ;applies stash and pops
        # ;garbage collects unreachable objs, old refs
            # pack folder
            # git gc
        # rebase -i <hash> ;parent of branch's first commit
            # ;squash - merge multiple commits of a branch as one commit
            # rebasing with squashing
    # merge default branch to feature branch for up-to-date changes
## 18. Wrap Up

### Self
# COMMIT is a SNAPSHOT, not patches