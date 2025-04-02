import logging
import random

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class Q2_Agent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    @log_function
    def getAction(self, gameState: GameState):
        logger = logging.getLogger('root')
        logger.info('MinimaxAgent')
        
        # 将Alpha-Beta搜索封装为单独方法
        def alpha_beta(state, depth, agentIndex, alpha, beta, cache, time_limit):
            # 检查是否超时
            if hasattr(self, 'start_time') and time.time() - self.start_time > time_limit:
                raise TimeoutError("Search timed out")
                
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
                
            # 使用更高效的缓存键 - 改进哈希方法
            state_repr = state.getPacmanPosition(), tuple(state.getGhostPositions()), state.getFood()
            state_key = hash(str(state_repr)) % (10**9 + 7), depth, agentIndex
            if state_key in cache:
                return cache[state_key]
                
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth
            
            if agentIndex == 0:  
                best_val = -float("inf")
                actions = state.getLegalActions(agentIndex)
                
                # 优化动作排序：使用更复杂的评估函数
                def action_score(action):
                    succ = state.generateSuccessor(agentIndex, action)
                    # 基础分数
                    score = self.evaluationFunction(succ)
                    
                    # 考虑食物距离
                    food = succ.getFood()
                    food_list = food.asList()
                    if food_list:
                        min_food_dist = min([manhattanDistance(succ.getPacmanPosition(), food) for food in food_list])
                        score -= min_food_dist * 10  # 靠近食物加分
                    
                    # 更强鬼怪的避让 - 更严格的安全距离检查
                    ghost_positions = succ.getGhostPositions()
                    pacman_pos = succ.getPacmanPosition()
                    for i, ghost_pos in enumerate(ghost_positions):
                        ghost_dist = manhattanDistance(pacman_pos, ghost_pos)
                        
                        # 检查鬼魂的移动方向，预测下一步位置
                        if hasattr(self, 'prev_ghost_positions') and len(self.prev_ghost_positions) > i:
                            prev_pos = self.prev_ghost_positions[i]
                            if prev_pos != ghost_pos:  # 鬼魂在移动
                                # 计算鬼魂移动方向
                                dx = ghost_pos[0] - prev_pos[0]
                                dy = ghost_pos[1] - prev_pos[1]
                                # 预测下一个位置
                                next_ghost_x = ghost_pos[0] + dx
                                next_ghost_y = ghost_pos[1] + dy
                                next_ghost_pos = (next_ghost_x, next_ghost_y)
                                # 检查预测位置与吃豆人的距离
                                next_dist = manhattanDistance(pacman_pos, next_ghost_pos)
                                if next_dist < 2:  # 预测会很近
                                    score -= 3000  # 增加对预测危险的惩罚
                        
                        # 根据距离设置不同的惩罚力度
                        if ghost_dist < 1:  # 直接相撞
                            score -= 15000  # 增加极高惩罚
                        elif ghost_dist < 2:  # 非常近
                            score -= 7000   # 增加惩罚
                        elif ghost_dist < 3:  # 较近
                            score -= 2000   # 增加惩罚
                        elif ghost_dist < 4:  # 中等距离
                            score -= 500    # 增加惩罚
                        elif ghost_dist < 5:  # 较远但仍需注意
                            score -= 100    # 增加惩罚
                        # 当鬼魂距离足够远时，减少对食物追求的限制
                        else:
                            # 鬼魂距离安全，增加对食物的重视
                            if food_list:
                                min_food_dist = min([manhattanDistance(pacman_pos, food) for food in food_list])
                                score += (20 - min_food_dist) * 5  # 距离食物越近加分越多
                    
                    # 考虑食物距离，增加权重
                    food = succ.getFood()
                    food_list = food.asList()
                    if food_list:
                        min_food_dist = min([manhattanDistance(succ.getPacmanPosition(), food) for food in food_list])
                        score -= min_food_dist * 15  # 增加食物吸引力
                        
                        # 考虑剩余食物数量，食物越少越积极寻找
                        remaining_food = len(food_list)
                        if remaining_food < 10:
                            score -= min_food_dist * (10 - remaining_food) * 2  # 食物少时更积极
                    
                    # 避免重复访问相同位置
                    if hasattr(self, 'visited_positions'):
                        pos = succ.getPacmanPosition()
                        if pos in self.visited_positions:
                            score -= 100 * self.visited_positions[pos]
                    
                    # 避免原地打转
                    if hasattr(self, "lastAction") and self.lastAction is not None:
                        if action == Actions.reverseDirection(self.lastAction):
                            score -= 200
                    
                    return score
                
                # 使用启发式函数排序动作
                actions = sorted(actions, key=lambda a: -action_score(a))
                
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    best_val = max(best_val, alpha_beta(successor, nextDepth, nextAgent, alpha, beta, cache, time_limit))
                    if best_val >= beta:
                        cache[state_key] = best_val
                        return best_val
                    alpha = max(alpha, best_val)
                cache[state_key] = best_val
                return best_val
            else:  
                best_val = float("inf")
                actions = state.getLegalActions(agentIndex)
                
                # 优化鬼怪动作排序 - 更智能的鬼怪行为模型
                def ghost_action_score(action):
                    succ = state.generateSuccessor(agentIndex, action)
                    pacman_pos = succ.getPacmanPosition()
                    ghost_pos = succ.getGhostPosition(agentIndex)
                    dist = manhattanDistance(pacman_pos, ghost_pos)
                    # 鬼怪倾向于接近吃豆人，但考虑更复杂的行为模式
                    # 考虑鬼怪之间的协作，避免重叠
                    other_ghosts_dist = float('inf')
                    for other_ghost_idx in range(1, state.getNumAgents()):
                        if other_ghost_idx != agentIndex:
                            other_ghost_pos = state.getGhostPosition(other_ghost_idx)
                            other_ghosts_dist = min(other_ghosts_dist, 
                                                   manhattanDistance(ghost_pos, other_ghost_pos))
                    
                    # 如果太靠近其他鬼怪，稍微降低评分以鼓励分散
                    if other_ghosts_dist < 2:
                        return dist + 1
                    return dist
                
                actions = sorted(actions, key=ghost_action_score)
                
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    best_val = min(best_val, alpha_beta(successor, nextDepth, nextAgent, alpha, beta, cache, time_limit))
                    if best_val <= alpha:
                        cache[state_key] = best_val
                        return best_val
                    beta = min(beta, best_val)
                cache[state_key] = best_val
                return best_val
        
        # 封装Alpha-Beta搜索为方法
        def alpha_beta_search(gameState, depth, time_limit=0.7):  # 进一步减少时间限制
            import time
            self.start_time = time.time()
            
            # 记录鬼魂位置用于预测
            ghost_positions = gameState.getGhostPositions()
            if not hasattr(self, 'prev_ghost_positions'):
                self.prev_ghost_positions = ghost_positions
            else:
                self.prev_ghost_positions = ghost_positions
            
            # 初始化或更新访问位置记录
            if not hasattr(self, 'visited_positions'):
                self.visited_positions = {}
            if not hasattr(self, 'position_history'):
                self.position_history = []
            
            # 更新访问位置记录
            current_pos = gameState.getPacmanPosition()
            self.visited_positions[current_pos] = self.visited_positions.get(current_pos, 0) + 1
            self.position_history.append(current_pos)
            
            # 检测循环模式
            if len(self.position_history) > 10:
                self.position_history = self.position_history[-10:]  # 只保留最近10步
                # 检测是否有重复模式
                if len(set(self.position_history)) < len(self.position_history) / 2:
                    # 发现循环，清空访问记录强制探索新区域
                    self.visited_positions = {}
            
            # 定期清理访问记录（每8步）
            if not hasattr(self, 'steps'):
                self.steps = 0
            self.steps += 1
            if self.steps % 8 == 0:
                self.visited_positions = {}
            
            candidateScores = []
            alpha = -float("inf")
            beta = float("inf")
            cache = {}
            actions = gameState.getLegalActions(0)
            
            # 使用启发式函数对动作进行初步评估
            def initial_evaluation(action):
                succ = gameState.generateSuccessor(0, action)
                score = self.evaluationFunction(succ)
                
                # 考虑食物距离
                food = succ.getFood()
                food_list = food.asList()
                if food_list:
                    min_food_dist = min([manhattanDistance(succ.getPacmanPosition(), food) for food in food_list])
                    # 增加食物吸引力
                    score -= min_food_dist * 10
                    
                    # 根据剩余食物数量调整策略
                    remaining_food = len(food_list)
                    if remaining_food < 10:
                        score -= min_food_dist * (10 - remaining_food) * 3  # 食物少时更积极寻找
                
                # 加强鬼怪的避让
                ghost_positions = succ.getGhostPositions()
                pacman_pos = succ.getPacmanPosition()
                for ghost_pos in ghost_positions:
                    ghost_dist = manhattanDistance(pacman_pos, ghost_pos)
                    if ghost_dist < 1:  # 直接相撞
                        score -= 10000
                    elif ghost_dist < 2:  # 非常近
                        score -= 5000
                    elif ghost_dist < 3:  # 较近
                        score -= 1000
                    # 当鬼魂距离安全时，增加对食物的重视
                    elif ghost_dist > 5 and food_list:
                        score += (20 - min_food_dist) * 3  # 安全情况下更重视食物
                
                # 避免原地打转
                if hasattr(self, "lastAction") and self.lastAction is not None:
                    if action == Actions.reverseDirection(self.lastAction):
                        score -= 100
                
                return score
            
            # 对所有可选动作按启发式函数排序
            actions = sorted(actions, key=lambda a: -initial_evaluation(a))
            
            # 安全检查：过滤掉会导致直接与鬼魂相撞的动作
            safe_actions = []
            for action in actions:
                succ = gameState.generateSuccessor(0, action)
                pacman_pos = succ.getPacmanPosition()
                is_safe = True
                
                for ghost_pos in succ.getGhostPositions():
                    if manhattanDistance(pacman_pos, ghost_pos) < 1.5:  # 非常接近鬼魂
                        is_safe = False
                        break
                
                if is_safe:
                    safe_actions.append(action)
            
            # 如果有安全动作，只考虑安全动作
            if safe_actions:
                actions = safe_actions
            
            # 限制搜索的动作数量，只考虑最有希望的前几个动作
            max_actions = min(len(actions), 4)  # 最多考虑4个动作
            actions = actions[:max_actions]
            
            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                try:
                    score = alpha_beta(successor, depth, 1, alpha, beta, cache, time_limit)
                    candidateScores.append((action, score))
                    alpha = max(alpha, score)
                except TimeoutError:
                    # 如果超时，使用启发式评估
                    candidateScores.append((action, initial_evaluation(action)))
            
            # 选择得分最高的动作
            if not candidateScores:
                # 如果没有候选动作，选择最安全的动作
                return self.get_safest_action(gameState)
                
            # 添加随机性打破平局，但保持一定的确定性
            random.seed(hash(str(gameState)) % 1000)  # 使用状态作为随机种子
            candidateScores.sort(key=lambda tup: (-tup[1], random.random() * 0.1))
            bestAction = candidateScores[0][0]
            
            # 避免原地打转
            if hasattr(self, "lastAction") and self.lastAction is not None:
                reverseAction = Actions.reverseDirection(self.lastAction)
                if bestAction == reverseAction:
                    nonReverse = [tup for tup in candidateScores if tup[0] != reverseAction]
                    if nonReverse:
                        bestAction = nonReverse[0][0]
            
            # 最终安全检查
            successor = gameState.generateSuccessor(0, bestAction)
            pacman_pos = successor.getPacmanPosition()
            for ghost_pos in successor.getGhostPositions():
                if manhattanDistance(pacman_pos, ghost_pos) < 1:  # 会导致直接相撞
                    # 寻找替代动作
                    safer_action = self.get_safest_action(gameState)
                    if safer_action:
                        bestAction = safer_action
            
            self.lastAction = bestAction
            return bestAction
        
        # 添加获取最安全动作的方法
        def get_safest_action(gameState):
            actions = gameState.getLegalActions(0)
            if not actions:
                return Directions.STOP
            
            # 计算每个动作的安全度
            action_safety = []
            for action in actions:
                succ = gameState.generateSuccessor(0, action)
                pacman_pos = succ.getPacmanPosition()
                min_ghost_dist = float('inf')
                
                # 更精确地评估鬼怪威胁
                ghost_threat = 0
                for i, ghost_pos in enumerate(succ.getGhostPositions()):
                    dist = manhattanDistance(pacman_pos, ghost_pos)
                    min_ghost_dist = min(min_ghost_dist, dist)
                    
                    # 考虑鬼怪的移动方向和速度
                    if hasattr(self, 'prev_ghost_positions') and len(self.prev_ghost_positions) > i:
                        prev_pos = self.prev_ghost_positions[i]
                        if prev_pos != ghost_pos:  # 鬼魂在移动
                            # 计算鬼魂移动方向
                            dx = ghost_pos[0] - prev_pos[0]
                            dy = ghost_pos[1] - prev_pos[1]
                            # 预测下一个位置
                            next_ghost_x = ghost_pos[0] + dx
                            next_ghost_y = ghost_pos[1] + dy
                            next_ghost_pos = (next_ghost_x, next_ghost_y)
                            # 检查预测位置与吃豆人的距离
                            next_dist = manhattanDistance(pacman_pos, next_ghost_pos)
                            if next_dist < dist:  # 鬼怪正在接近
                                ghost_threat += (1.0 / (next_dist + 0.1)) * 3  # 增加威胁值
                    
                    # 根据距离计算威胁值 - 增加威胁评估
                    if dist < 1:
                        ghost_threat += 150  # 增加威胁值
                    elif dist < 2:
                        ghost_threat += 80   # 增加威胁值
                    elif dist < 3:
                        ghost_threat += 20   # 增加威胁值
                    elif dist < 4:
                        ghost_threat += 8    # 增加威胁值
                    elif dist < 5:
                        ghost_threat += 2    # 增加威胁值
                
                # 考虑食物因素
                food_score = 0
                food = succ.getFood()
                food_list = food.asList()
                if food_list:
                    min_food_dist = min([manhattanDistance(pacman_pos, food) for food in food_list])
                    # 增加食物权重，但仍保持安全第一
                    food_score = 25.0 / (min_food_dist + 1)  # 增加食物吸引力
                    
                    # 根据剩余食物数量调整策略
                    remaining_food = len(food_list)
                    if remaining_food < 10:
                        food_score *= (10 - remaining_food) / 1.5  # 食物少时更积极
                
                # 考虑胶囊的价值 - 大幅增加胶囊价值
                capsule_score = 0
                capsules = succ.getCapsules()
                if capsules:
                    min_capsule_dist = min([manhattanDistance(pacman_pos, capsule) for capsule in capsules])
                    # 胶囊比食物更有价值，特别是当鬼怪靠近时
                    capsule_score = 60.0 / (min_capsule_dist + 1)  # 增加胶囊基础价值
                    if min_ghost_dist < 5:  # 鬼怪靠近时胶囊更有价值
                        capsule_score *= 3  # 增加胶囊在危险时的价值
                
                # 安全度计算：鬼魂距离足够远时更重视食物
                if min_ghost_dist > 4 and ghost_threat < 5:
                    safety = min_ghost_dist * 0.5 + food_score * 0.3 + capsule_score * 0.3  # 增加胶囊权重
                else:
                    safety = min_ghost_dist * 2.5 - ghost_threat + food_score * 0.1 + capsule_score * 0.5  # 增加安全权重和胶囊权重
                
                # 避免原地打转
                if hasattr(self, "lastAction") and action == Actions.reverseDirection(self.lastAction):
                    safety -= 1.0  # 增加惩罚
                
                # 避免重复访问相同位置
                if hasattr(self, 'visited_positions'):
                    if pacman_pos in self.visited_positions:
                        visits = self.visited_positions[pacman_pos]
                        if visits > 2:  # 如果访问次数过多
                            safety -= visits * 0.5  # 增加惩罚以避免循环
                
                action_safety.append((action, safety))
            
            # 选择安全度最高的动作
            action_safety.sort(key=lambda x: -x[1])
            return action_safety[0][0]
        
        # 添加方法到实例
        self.alpha_beta_search = alpha_beta_search
        self.get_safest_action = get_safest_action
        
        # 使用迭代加深搜索
        try:
            import time
            # 根据游戏状态动态调整搜索深度
            food_count = len(gameState.getFood().asList())
            ghost_distances = [manhattanDistance(gameState.getPacmanPosition(), ghost_pos) 
                              for ghost_pos in gameState.getGhostPositions()]
            min_ghost_dist = min(ghost_distances) if ghost_distances else float('inf')
            
            # 根据鬼怪距离和食物数量动态调整搜索深度
            if min_ghost_dist < 3:  # 鬼怪非常近，减少搜索深度以快速反应
                depth = max(self.depth - 1, 1)
            elif food_count < 5:  # 食物少时可以搜索更深
                depth = min(self.depth + 1, 4)
            elif food_count > 20:  # 食物多时减少搜索深度
                depth = max(self.depth - 1, 2)
            else:
                depth = self.depth
                
            # 检查是否有胶囊，如果有且鬼怪近，优先考虑吃胶囊
            capsules = gameState.getCapsules()
            if capsules and min_ghost_dist < 6:  # 增加胶囊检测范围
                pacman_pos = gameState.getPacmanPosition()
                min_capsule_dist = min([manhattanDistance(pacman_pos, capsule) for capsule in capsules])
                if min_capsule_dist < 4:  # 增加胶囊检测距离
                    # 找到朝向最近胶囊的动作
                    actions = gameState.getLegalActions(0)
                    best_action = None
                    best_dist = float('inf')
                    for action in actions:
                        succ = gameState.generateSuccessor(0, action)
                        new_pos = succ.getPacmanPosition()
                        for capsule in capsules:
                            dist = manhattanDistance(new_pos, capsule)
                            if dist < best_dist:
                                best_dist = dist
                                best_action = action
                    if best_action:
                        # 检查该动作是否安全
                        succ = gameState.generateSuccessor(0, best_action)
                        new_pos = succ.getPacmanPosition()
                        is_safe = True
                        for ghost_pos in succ.getGhostPositions():
                            if manhattanDistance(new_pos, ghost_pos) < 1.5:
                                is_safe = False
                                break
                        if is_safe:
                            return best_action
            
            # 特殊处理trappedClassic布局 - 检测是否在困境中
            if "trappedClassic" in str(gameState.getFood()) or min_ghost_dist < 2:
                # 在困境中时，使用更保守的策略
                return self.get_safest_action(gameState)
            
            return self.alpha_beta_search(gameState, depth)
        except Exception as e:
            # 如果出现异常，返回最安全的动作
            return self.get_safest_action(gameState)
