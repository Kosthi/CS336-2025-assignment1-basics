import torch
import torch.nn.functional as F


def decode(
    model: torch.nn.Module,
    tokenizer,
    prompt: str | list[int],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str = "cpu",
) -> str:
    """
    从语言模型生成文本完成。

    参数:
        model: 语言模型
        tokenizer: 分词器，需要有encode和decode方法
        prompt: 输入提示（字符串或token IDs列表）
        max_new_tokens: 最大生成token数量
        temperature: 温度参数（>0）。值越高越随机，值越低越确定
        top_p: nucleus采样阈值（0.0-1.0）。只从累积概率最高的token中采样
        eos_token_id: 结束token的ID，如果为None则不会基于结束token停止
        device: 设备

    返回:
        str: 生成的文本
    """
    # 确保模型在评估模式
    model.eval()
    model.to(device)

    # 处理输入prompt
    if isinstance(prompt, str):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    else:
        input_ids = torch.tensor([prompt], dtype=torch.long).to(device)

    # 如果提供了eos_token_id但不在tokenizer中，使用默认值
    if eos_token_id is None and hasattr(tokenizer, "eos_token_id"):
        eos_token_id = tokenizer.eos_token_id

    # 生成序列
    generated_ids = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 前向传播
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)

            # 获取最后一个token的logits
            next_token_logits = outputs.logits[:, -1, :]

            # 应用温度缩放
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 应用top-p（nucleus）采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p

                # 确保至少保留一个token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # 创建要移除token的mask
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

            # 计算概率
            probs = F.softmax(next_token_logits, dim=-1)

            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)

            # 检查是否生成了结束token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            # 将新token添加到生成序列
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # 更新attention mask
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)

    # 解码为文本
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    return generated_text


def generate_completions(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>",
    device: str = "cpu",
) -> list[str]:
    """
    为多个提示生成补全。

    参数:
        model: 语言模型
        tokenizer: 分词器
        prompts: 提示列表
        max_new_tokens: 最大生成token数量
        temperature: 温度参数
        top_p: nucleus采样阈值
        eos_token: 结束token字符串
        device: 设备

    返回:
        List[str]: 生成的文本列表
    """
    # 获取结束token的ID
    eos_token_id = tokenizer.encode(eos_token)[0] if eos_token in tokenizer.get_vocab() else None

    completions = []
    for prompt in prompts:
        completion = decode(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            device=device,
        )
        completions.append(completion)

    return completions


# 更高级的版本，支持批量生成和更多控制选项
class Decoder:
    """高级解码器，支持更多功能和选项"""

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # 将模型移动到指定设备并设置为评估模式
        self.model.to(device)
        self.model.eval()

        # 获取结束token ID
        self.eos_token_id = None
        if hasattr(tokenizer, "eos_token"):
            if tokenizer.eos_token is not None:
                self.eos_token_id = tokenizer.encode(tokenizer.eos_token)[0]

    def _apply_temperature(self, logits, temperature):
        """应用温度缩放"""
        if temperature != 1.0:
            logits = logits / temperature
        return logits

    def _apply_top_p(self, logits, top_p):
        """应用top-p（nucleus）采样"""
        if top_p >= 1.0:
            return logits

        # 对logits进行排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # 移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p

        # 确保至少保留一个token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 创建要移除token的mask
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        # 将移除的token的logits设置为负无穷
        filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))

        return filtered_logits

    def _apply_top_k(self, logits, top_k):
        """应用top-k采样"""
        if top_k <= 0:
            return logits

        # 获取top-k个最大的logits
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))

        # 创建mask：只保留top-k的token
        min_top_k = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, torch.tensor(float("-inf")).to(logits.device), logits)

        return logits

    def _sample_next_token(self, logits, temperature=1.0, top_p=1.0, top_k=0):
        """采样下一个token"""
        # 应用温度
        logits = self._apply_temperature(logits, temperature)

        # 应用top-k（如果需要）
        if top_k > 0:
            logits = self._apply_top_k(logits, top_k)

        # 应用top-p
        logits = self._apply_top_p(logits, top_p)

        # 计算概率
        probs = F.softmax(logits, dim=-1)

        # 采样
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def decode(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        stop_strings: list[str] | None = None,
        return_tokens: bool = False,
    ) -> str | list[int]:
        """
        从模型生成文本。

        参数:
            prompt: 输入提示
            max_new_tokens: 最大生成token数量
            temperature: 温度参数
            top_p: nucleus采样阈值
            top_k: top-k采样参数
            repetition_penalty: 重复惩罚
            stop_strings: 停止字符串列表
            return_tokens: 是否返回token IDs

        返回:
            生成的文本或token IDs
        """
        # 编码输入
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            input_ids = torch.tensor([prompt], dtype=torch.long).to(self.device)

        generated_ids = input_ids.clone()

        # 创建attention mask
        attention_mask = torch.ones_like(input_ids)

        # 用于重复惩罚
        generated_tokens_set = set()

        with torch.no_grad():
            for i in range(max_new_tokens):
                # 前向传播
                outputs = self.model(input_ids=generated_ids, attention_mask=attention_mask)

                # 获取最后一个token的logits
                next_token_logits = outputs.logits[:, -1, :]
                if repetition_penalty != 1.0:
                    next_token_logits = next_token_logits.clone()
                    for token_id in generated_tokens_set:
                        if token_id < next_token_logits.shape[-1]:
                            next_token_logits[:, token_id] /= repetition_penalty

                # 采样下一个token
                next_token = self._sample_next_token(
                    next_token_logits, temperature=temperature, top_p=top_p, top_k=top_k
                )

                # 添加到已生成token集合
                generated_tokens_set.add(next_token.item())

                # 检查是否生成了结束token
                if self.eos_token_id is not None and next_token.item() == self.eos_token_id:
                    break

                # 添加到生成序列
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # 更新attention mask
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=torch.long, device=self.device)], dim=1
                )

                # 检查停止字符串
                if stop_strings:
                    current_text = self.tokenizer.decode(generated_ids[0].tolist())
                    for stop_str in stop_strings:
                        if stop_str in current_text:
                            # 返回停止字符串之前的内容
                            stop_index = current_text.find(stop_str)
                            if stop_index != -1:
                                generated_ids = torch.tensor([self.tokenizer.encode(current_text[:stop_index])]).to(
                                    self.device
                                )
                                break
                    else:
                        continue
                    break

        # 解码或返回token IDs
        if return_tokens:
            return generated_ids[0].tolist()
        else:
            return self.tokenizer.decode(generated_ids[0].tolist())

    def batch_decode(
        self,
        prompts: list[str | list[int]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> list[str]:
        """批量解码"""
        results = []
        for prompt in prompts:
            result = self.decode(
                prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, **kwargs
            )
            results.append(result)
        return results


# 简单的适配器函数
def run_decode(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>",
) -> str:
    """
    适配器函数，用于测试。
    """
    return decode(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.encode(eos_token)[0] if eos_token in tokenizer.get_vocab() else None,
    )
